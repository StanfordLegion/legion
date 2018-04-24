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

#include "legion/legion_config.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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
  NEW_OPAQUE_TYPE(legion_domain_point_iterator_t);
  NEW_OPAQUE_TYPE(legion_coloring_t);
  NEW_OPAQUE_TYPE(legion_domain_coloring_t);
  NEW_OPAQUE_TYPE(legion_point_coloring_t);
  NEW_OPAQUE_TYPE(legion_domain_point_coloring_t);
  NEW_OPAQUE_TYPE(legion_multi_domain_point_coloring_t);
  NEW_OPAQUE_TYPE(legion_index_space_allocator_t);
  NEW_OPAQUE_TYPE(legion_field_allocator_t);
  NEW_OPAQUE_TYPE(legion_argument_map_t);
  NEW_OPAQUE_TYPE(legion_predicate_t);
  NEW_OPAQUE_TYPE(legion_future_t);
  NEW_OPAQUE_TYPE(legion_future_map_t);
  NEW_OPAQUE_TYPE(legion_task_launcher_t);
  NEW_OPAQUE_TYPE(legion_index_launcher_t);
  NEW_OPAQUE_TYPE(legion_inline_launcher_t);
  NEW_OPAQUE_TYPE(legion_copy_launcher_t);
  NEW_OPAQUE_TYPE(legion_index_copy_launcher_t);
  NEW_OPAQUE_TYPE(legion_acquire_launcher_t);
  NEW_OPAQUE_TYPE(legion_release_launcher_t);
  NEW_OPAQUE_TYPE(legion_attach_launcher_t);
  NEW_OPAQUE_TYPE(legion_must_epoch_launcher_t);
  NEW_OPAQUE_TYPE(legion_physical_region_t);
  NEW_OPAQUE_TYPE(legion_accessor_array_1d_t);
  NEW_OPAQUE_TYPE(legion_accessor_array_2d_t);
  NEW_OPAQUE_TYPE(legion_accessor_array_3d_t);
  NEW_OPAQUE_TYPE(legion_index_iterator_t);
  NEW_OPAQUE_TYPE(legion_task_t);
  NEW_OPAQUE_TYPE(legion_inline_t);
  NEW_OPAQUE_TYPE(legion_mappable_t);
  NEW_OPAQUE_TYPE(legion_region_requirement_t);
  NEW_OPAQUE_TYPE(legion_machine_t);
  NEW_OPAQUE_TYPE(legion_mapper_t);
  NEW_OPAQUE_TYPE(legion_default_mapper_t);
  NEW_OPAQUE_TYPE(legion_processor_query_t);
  NEW_OPAQUE_TYPE(legion_memory_query_t);
  NEW_OPAQUE_TYPE(legion_machine_query_interface_t);
  NEW_OPAQUE_TYPE(legion_execution_constraint_set_t);
  NEW_OPAQUE_TYPE(legion_layout_constraint_set_t);
  NEW_OPAQUE_TYPE(legion_task_layout_constraint_set_t);
  NEW_OPAQUE_TYPE(legion_slice_task_output_t);
  NEW_OPAQUE_TYPE(legion_map_task_input_t);
  NEW_OPAQUE_TYPE(legion_map_task_output_t);
  NEW_OPAQUE_TYPE(legion_physical_instance_t);
  NEW_OPAQUE_TYPE(legion_mapper_runtime_t);
  NEW_OPAQUE_TYPE(legion_mapper_context_t);
  NEW_OPAQUE_TYPE(legion_field_map_t);
#undef NEW_OPAQUE_TYPE

  /**
   * @see ptr_t
   */
  typedef struct legion_ptr_t {
    long long int value;
  } legion_ptr_t;

  typedef legion_coord_t coord_t;

#define NEW_POINT_TYPE(T, DIM) typedef struct T { coord_t x[DIM]; } T
  NEW_POINT_TYPE(legion_point_1d_t, 1);
  NEW_POINT_TYPE(legion_point_2d_t, 2);
  NEW_POINT_TYPE(legion_point_3d_t, 3);
#undef NEW_POINT_TYPE

#define NEW_RECT_TYPE(T, PT) typedef struct T { PT lo, hi; } T
  NEW_RECT_TYPE(legion_rect_1d_t, legion_point_1d_t);
  NEW_RECT_TYPE(legion_rect_2d_t, legion_point_2d_t);
  NEW_RECT_TYPE(legion_rect_3d_t, legion_point_3d_t);
#undef NEW_RECT_TYPE

#define NEW_BLOCKIFY_TYPE(T, PT) \
  typedef struct T { PT block_size; PT offset; } T
  NEW_BLOCKIFY_TYPE(legion_blockify_1d_t, legion_point_1d_t);
  NEW_BLOCKIFY_TYPE(legion_blockify_2d_t, legion_point_2d_t);
  NEW_BLOCKIFY_TYPE(legion_blockify_3d_t, legion_point_3d_t);
#undef NEW_BLOCKIFY_TYPE

#define NEW_TRANSFORM_TYPE(T, X, Y) \
  typedef struct T { coord_t trans[X][Y]; } T
  NEW_TRANSFORM_TYPE(legion_transform_1x1_t, 1, 1);
  NEW_TRANSFORM_TYPE(legion_transform_1x2_t, 1, 2);
  NEW_TRANSFORM_TYPE(legion_transform_1x3_t, 1, 3);
  NEW_TRANSFORM_TYPE(legion_transform_2x1_t, 2, 1);
  NEW_TRANSFORM_TYPE(legion_transform_2x2_t, 2, 2);
  NEW_TRANSFORM_TYPE(legion_transform_2x3_t, 2, 3);
  NEW_TRANSFORM_TYPE(legion_transform_3x1_t, 3, 1);
  NEW_TRANSFORM_TYPE(legion_transform_3x2_t, 3, 2);
  NEW_TRANSFORM_TYPE(legion_transform_3x3_t, 3, 3);
#undef NEW_TRANSFORM_TYPE

#define NEW_AFFINE_TRANSFORM_TYPE(T, TT, PT) \
  typedef struct T { TT transform; PT offset; } T
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_1x1_t, 
                            legion_transform_1x1_t, legion_point_1d_t);
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_1x2_t,
                            legion_transform_1x2_t, legion_point_1d_t);
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_1x3_t,
                            legion_transform_1x3_t, legion_point_1d_t);
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_2x1_t, 
                            legion_transform_2x1_t, legion_point_2d_t);
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_2x2_t,
                            legion_transform_2x2_t, legion_point_2d_t);
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_2x3_t,
                            legion_transform_2x3_t, legion_point_2d_t);
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_3x1_t, 
                            legion_transform_3x1_t, legion_point_3d_t);
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_3x2_t,
                            legion_transform_3x2_t, legion_point_3d_t);
  NEW_AFFINE_TRANSFORM_TYPE(legion_affine_transform_3x3_t,
                            legion_transform_3x3_t, legion_point_3d_t);
#undef NEW_AFFINE_TRANSFORM_TYPE

  /**
   * @see Legion::Domain
   */
  typedef struct legion_domain_t {
    realm_id_t is_id;
    int dim;
// Hack: Python CFFI isn't smart enough to do constant folding so we
// have to do this by hand here. To avoid this bitrotting, at least
// make the preprocessor check that the value is equal to what we
// expect.
#define MAX_DOMAIN_DIM 6 // 2 * LEGION_MAX_RECT_DIM
#if MAX_DOMAIN_DIM != 2 * LEGION_MAX_RECT_DIM // sanity check value
#error Mismatch in MAX_DOMAIN_DIM
#endif
    coord_t rect_data[MAX_DOMAIN_DIM];
#undef MAX_DOMAIN_DIM
  } legion_domain_t;

  /**
   * @see Legion::DomainPoint
   */
  typedef struct legion_domain_point_t {
    int dim;
    coord_t point_data[MAX_POINT_DIM];
  } legion_domain_point_t;

  /**
   * @see Legion::Transform
   */
  typedef struct legion_domain_transform_t {
    int m, n;
// Hack: Python CFFI isn't smart enough to do constant folding so we
// have to do this by hand here. To avoid this bitrotting, at least
// make the preprocessor check that the value is equal to what we
// expect.
#define MAX_MATRIX_DIM 9
#if MAX_MATRIX_DIM != LEGION_MAX_POINT_DIM * LEGION_MAX_POINT_DIM // sanity check
#error Mismatch in MAX_MATRIX_DIM
#endif
    coord_t matrix[MAX_MATRIX_DIM];
#undef MAX_MATRIX_DIM
  } legion_domain_transform_t;

  /**
   * @see Legion::DomainAffineTransform
   */
  typedef struct legion_domain_affine_transform_t {
    legion_domain_transform_t transform;
    legion_domain_point_t offset;
  } legion_domain_affine_transform_t;

  /**
   * @see Legion::IndexSpace
   */
  typedef struct legion_index_space_t {
    legion_index_space_id_t id;
    legion_index_tree_id_t tid;
    legion_type_tag_t type_tag;
  } legion_index_space_t;

  /**
   * @see Legion::IndexPartition
   */
  typedef struct legion_index_partition_t {
    legion_index_partition_id_t id;
    legion_index_tree_id_t tid;
    legion_type_tag_t type_tag;
  } legion_index_partition_t;

  /**
   * @see Legion::FieldSpace
   */
  typedef struct legion_field_space_t {
    legion_field_space_id_t id;
  } legion_field_space_t;

  /**
   * @see Legion::LogicalRegion
   */
  typedef struct legion_logical_region_t {
    legion_region_tree_id_t tree_id;
    legion_index_space_t index_space;
    legion_field_space_t field_space;
  } legion_logical_region_t;

  /**
   * @see Legion::LogicalPartition
   */
  typedef struct legion_logical_partition_t {
    legion_region_tree_id_t tree_id;
    legion_index_partition_t index_partition;
    legion_field_space_t field_space;
  } legion_logical_partition_t;

  /**
   * @see Legion::TaskArgument
   */
  typedef struct legion_task_argument_t {
    void *args;
    size_t arglen;
  } legion_task_argument_t;

  typedef struct legion_byte_offset_t {
    int offset;
  } legion_byte_offset_t;

  /**
   * @see Legion::InputArgs
   */
  typedef struct legion_input_args_t {
    char **argv;
    int argc;
  } legion_input_args_t;

  /**
   * @see Legion::TaskConfigOptions
   */
  typedef struct legion_task_config_options_t {
    bool leaf /* = false */;
    bool inner /* = false */;
    bool idempotent /* = false */;
  }  legion_task_config_options_t;

  /**
   * @see Legion::Processor
   */
  typedef struct legion_processor_t {
    realm_id_t id;
  } legion_processor_t;

  /**
   * @see Legion::Memory
   */
  typedef struct legion_memory_t {
    realm_id_t id;
  } legion_memory_t;

  /**
   * @see Legion::Mapper::TaskSlice
   */
  typedef struct legion_task_slice_t {
    legion_domain_t domain;
    legion_processor_t proc;
    bool recurse;
    bool stealable;
  } legion_task_slice_t;

  /**
   * @see Legion::PhaseBarrier
   */
  typedef struct legion_phase_barrier_t {
    // From Realm::Event
    realm_id_t id;
    // From Realm::Barrier
    realm_barrier_timestamp_t timestamp;
  } legion_phase_barrier_t;

  /**
   * @see Legion::DynamicCollective
   */
  typedef struct legion_dynamic_collective_t {
    // From Legion::PhaseBarrier
    //   From Realm::Event
    realm_id_t id;
    //   From Realm::Barrier
    realm_barrier_timestamp_t timestamp;
    // From Legion::DynamicCollective
    legion_reduction_op_id_t redop;
  } legion_dynamic_collective_t;

  /**
   * @see Legion::Mapping::Mapper::TaskOptions
   */
  typedef struct legion_task_options_t {
    legion_processor_t initial_proc;
    bool inline_task;
    bool stealable;
    bool map_locally;
    legion_task_priority_t parent_priority;
  } legion_task_options_t;

  typedef struct legion_slice_task_input_t {
    legion_domain_t domain;
  } legion_slice_task_input_t;

  /**
   * Interface for a Legion C registration callback.
   */
  typedef
    void (*legion_registration_callback_pointer_t)(
      legion_machine_t /* machine */,
      legion_runtime_t /* runtime */,
      const legion_processor_t * /* local_procs */,
      unsigned /* num_local_procs */);

  /**
   * Interface for a Legion C task that is wrapped (i.e. this is the Realm
   * task interface)
   */
  typedef realm_task_pointer_t legion_task_pointer_wrapped_t;

  /**
   * Interface for a Legion C projection functor (Logical Region
   * upper bound).
   */
  typedef
    legion_logical_region_t (*legion_projection_functor_logical_region_t)(
      legion_runtime_t /* runtime */,
      const legion_mappable_t /* mappable */,
      unsigned /* index */,
      legion_logical_region_t /* upper_bound */,
      legion_domain_point_t /* point */);

  /**
   * Interface for a Legion C projection functor (Logical Partition
   * upper bound).
   */
  typedef
    legion_logical_region_t (*legion_projection_functor_logical_partition_t)(
      legion_runtime_t /* runtime */,
      const legion_mappable_t /* mappable */,
      unsigned /* index */,
      legion_logical_partition_t /* upper_bound */,
      legion_domain_point_t /* point */);

  // -----------------------------------------------------------------------
  // Pointer Operations
  // -----------------------------------------------------------------------

  /**
   * @see ptr_t::nil()
   */
  legion_ptr_t
  legion_ptr_nil(void);

  /**
   * @see ptr_t::is_null()
   */
  bool
  legion_ptr_is_null(legion_ptr_t ptr);

  /**
   * @see Legion::Runtime::safe_cast(
   *        Context, ptr_t, LogicalRegion)
   */
  legion_ptr_t
  legion_ptr_safe_cast(legion_runtime_t runtime,
                       legion_context_t ctx,
                       legion_ptr_t pointer,
                       legion_logical_region_t region);

  // -----------------------------------------------------------------------
  // Domain Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Domain::from_rect()
   */
  legion_domain_t
  legion_domain_from_rect_1d(legion_rect_1d_t r);

  /**
   * @see Legion::Domain::from_rect()
   */
  legion_domain_t
  legion_domain_from_rect_2d(legion_rect_2d_t r);

  /**
   * @see Legion::Domain::from_rect()
   */
  legion_domain_t
  legion_domain_from_rect_3d(legion_rect_3d_t r);

  /**
   * @see Legion::Domain::Domain(Legion::IndexSpace)
   */
  legion_domain_t
  legion_domain_from_index_space(legion_runtime_t runtime,
                                 legion_index_space_t is);

  /**
   * @see Legion::Domain::get_rect()
   */
  legion_rect_1d_t
  legion_domain_get_rect_1d(legion_domain_t d);

  /**
   * @see Legion::Domain::get_rect()
   */
  legion_rect_2d_t
  legion_domain_get_rect_2d(legion_domain_t d);

  /**
   * @see Legion::Domain::get_rect()
   */
  legion_rect_3d_t
  legion_domain_get_rect_3d(legion_domain_t d);

  bool
  legion_domain_is_dense(legion_domain_t d);

  // These are the same as above but will ignore 
  // the existence of any sparsity map, whereas the 
  // ones above will fail if a sparsity map exists
  legion_rect_1d_t
  legion_domain_get_bounds_1d(legion_domain_t d);

  legion_rect_2d_t
  legion_domain_get_bounds_2d(legion_domain_t d);

  legion_rect_3d_t
  legion_domain_get_bounds_3d(legion_domain_t d);

  /**
   * @see Legion::Domain::get_volume()
   */
  size_t
  legion_domain_get_volume(legion_domain_t d);

  // -----------------------------------------------------------------------
  // Domain Transform Operations
  // -----------------------------------------------------------------------

  legion_domain_transform_t
  legion_domain_transform_from_1x1(legion_transform_1x1_t t);

  legion_domain_transform_t
  legion_domain_transform_from_1x2(legion_transform_1x2_t t);

  legion_domain_transform_t
  legion_domain_transform_from_1x3(legion_transform_1x3_t t);

  legion_domain_transform_t
  legion_domain_transform_from_2x1(legion_transform_2x1_t t);

  legion_domain_transform_t
  legion_domain_transform_from_2x2(legion_transform_2x2_t t);

  legion_domain_transform_t
  legion_domain_transform_from_2x3(legion_transform_2x3_t t);

  legion_domain_transform_t
  legion_domain_transform_from_3x1(legion_transform_3x1_t t);

  legion_domain_transform_t
  legion_domain_transform_from_3x2(legion_transform_3x2_t t);

  legion_domain_transform_t
  legion_domain_transform_from_3x3(legion_transform_3x3_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_1x1(legion_affine_transform_1x1_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_1x2(legion_affine_transform_1x2_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_1x3(legion_affine_transform_1x3_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_2x1(legion_affine_transform_2x1_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_2x2(legion_affine_transform_2x2_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_2x3(legion_affine_transform_2x3_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_3x1(legion_affine_transform_3x1_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_3x2(legion_affine_transform_3x2_t t);

  legion_domain_affine_transform_t
  legion_domain_affine_transform_from_3x3(legion_affine_transform_3x3_t t);

  // -----------------------------------------------------------------------
  // Domain Point Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::DomainPoint::from_point()
   */
  legion_domain_point_t
  legion_domain_point_from_point_1d(legion_point_1d_t p);

  /**
   * @see Legion::DomainPoint::from_point()
   */
  legion_domain_point_t
  legion_domain_point_from_point_2d(legion_point_2d_t p);

  /**
   * @see Legion::DomainPoint::from_point()
   */
  legion_domain_point_t
  legion_domain_point_from_point_3d(legion_point_3d_t p);

  /**
   * @see Legion::DomainPoint::get_point()
   */
  legion_point_1d_t
  legion_domain_point_get_point_1d(legion_domain_point_t p);

  /**
   * @see Legion::DomainPoint::get_point()
   */
  legion_point_2d_t
  legion_domain_point_get_point_2d(legion_domain_point_t p);

  /**
   * @see Legion::DomainPoint::get_point()
   */
  legion_point_3d_t
  legion_domain_point_get_point_3d(legion_domain_point_t p);

  /**
   * @see Legion::DomainPoint::nil()
   */
  legion_domain_point_t
  legion_domain_point_nil();

  /**
   * @see Legion::DomainPoint::is_null()
   */
  bool
  legion_domain_point_is_null(legion_domain_point_t point);

  /**
   * @see Legion::Runtime::safe_cast(
   *        Context, DomainPoint, LogicalRegion)
   */
  legion_domain_point_t
  legion_domain_point_safe_cast(legion_runtime_t runtime,
                                legion_context_t ctx,
                                legion_domain_point_t point,
                                legion_logical_region_t region);

  // -----------------------------------------------------------------------
  // Domain Point Iterator
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Domain::DomainPointIterator::DomainPointIterator()
   */
  legion_domain_point_iterator_t
  legion_domain_point_iterator_create(legion_domain_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Domain::DomainPointIterator::~DomainPointIterator()
   */
  void
  legion_domain_point_iterator_destroy(legion_domain_point_iterator_t handle);

  /**
   * @see Legion::Domain::DomainPointIterator::any_left
   */
  bool
  legion_domain_point_iterator_has_next(legion_domain_point_iterator_t handle);

  /**
   * @see Legion::Domain::DomainPointIterator::step()
   */
  legion_domain_point_t
  legion_domain_point_iterator_next(legion_domain_point_iterator_t handle);

  // -----------------------------------------------------------------------
  // Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Coloring
   */
  legion_coloring_t
  legion_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Coloring
   */
  void
  legion_coloring_destroy(legion_coloring_t handle);

  /**
   * @see Legion::Coloring
   */
  void
  legion_coloring_ensure_color(legion_coloring_t handle,
                               legion_color_t color);

  /**
   * @see Legion::Coloring
   */
  void
  legion_coloring_add_point(legion_coloring_t handle,
                            legion_color_t color,
                            legion_ptr_t point);

  /**
   * @see Legion::Coloring
   */
  void
  legion_coloring_delete_point(legion_coloring_t handle,
                               legion_color_t color,
                               legion_ptr_t point);

  /**
   * @see Legion::Coloring
   */
  bool
  legion_coloring_has_point(legion_coloring_t handle,
                            legion_color_t color,
                            legion_ptr_t point);

  /**
   * @see Legion::Coloring
   */
  void
  legion_coloring_add_range(legion_coloring_t handle,
                            legion_color_t color,
                            legion_ptr_t start,
                            legion_ptr_t end /**< inclusive */);

  // -----------------------------------------------------------------------
  // Domain Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::DomainColoring
   */
  legion_domain_coloring_t
  legion_domain_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::DomainColoring
   */
  void
  legion_domain_coloring_destroy(legion_domain_coloring_t handle);

  /**
   * @see Legion::DomainColoring
   */
  void
  legion_domain_coloring_color_domain(legion_domain_coloring_t handle,
                                      legion_color_t color,
                                      legion_domain_t domain);

  /**
   * @see Legion::DomainColoring
   */
  legion_domain_t
  legion_domain_coloring_get_color_space(legion_domain_coloring_t handle);

  // -----------------------------------------------------------------------
  // Point Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::PointColoring
   */
  legion_point_coloring_t
  legion_point_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::PointColoring
   */
  void
  legion_point_coloring_destroy(
    legion_point_coloring_t handle);

  /**
   * @see Legion::PointColoring
   */
  void
  legion_point_coloring_add_point(legion_point_coloring_t handle,
                                  legion_domain_point_t color,
                                  legion_ptr_t point);

  /**
   * @see Legion::PointColoring
   */
  void
  legion_point_coloring_add_range(legion_point_coloring_t handle,
                                  legion_domain_point_t color,
                                  legion_ptr_t start,
                                  legion_ptr_t end /**< inclusive */);

  // -----------------------------------------------------------------------
  // Domain Point Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::DomainPointColoring
   */
  legion_domain_point_coloring_t
  legion_domain_point_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::DomainPointColoring
   */
  void
  legion_domain_point_coloring_destroy(
    legion_domain_point_coloring_t handle);

  /**
   * @see Legion::DomainPointColoring
   */
  void
  legion_domain_point_coloring_color_domain(
    legion_domain_point_coloring_t handle,
    legion_domain_point_t color,
    legion_domain_t domain);

  // -----------------------------------------------------------------------
  // Multi-Domain Point Coloring Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::MultiDomainPointColoring
   */
  legion_multi_domain_point_coloring_t
  legion_multi_domain_point_coloring_create(void);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::MultiDomainPointColoring
   */
  void
  legion_multi_domain_point_coloring_destroy(
    legion_multi_domain_point_coloring_t handle);

  /**
   * @see Legion::MultiDomainPointColoring
   */
  void
  legion_multi_domain_point_coloring_color_domain(
    legion_multi_domain_point_coloring_t handle,
    legion_domain_point_t color,
    legion_domain_t domain);

  // -----------------------------------------------------------------------
  // Index Space Operations
  // ----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_space(Context, size_t)
   */
  legion_index_space_t
  legion_index_space_create(legion_runtime_t runtime,
                            legion_context_t ctx,
                            size_t max_num_elmts);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_space(Context, Domain)
   */
  legion_index_space_t
  legion_index_space_create_domain(legion_runtime_t runtime,
                                   legion_context_t ctx,
                                   legion_domain_t domain);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::union_index_spaces
   */
  legion_index_space_t
  legion_index_space_union(legion_runtime_t runtime,
                           legion_context_t ctx,
                           const legion_index_space_t *spaces,
                           size_t num_spaces);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::intersect_index_spaces
   */
  legion_index_space_t
  legion_index_space_intersection(legion_runtime_t runtime,
                                  legion_context_t ctx,
                                  const legion_index_space_t *spaces,
                                  size_t num_spaces);

  /**
   * @see Legion::Runtime::has_multiple_domains().
   */
  bool
  legion_index_space_has_multiple_domains(legion_runtime_t runtime,
                                          legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::get_index_space_domain()
   */
  legion_domain_t
  legion_index_space_get_domain(legion_runtime_t runtime,
                                legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::get_parent_index_partition()
   */
  legion_index_partition_t
  legion_index_space_get_parent_index_partition(legion_runtime_t runtime,
                                                legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_index_space()
   */
  void
  legion_index_space_destroy(legion_runtime_t runtime,
                             legion_context_t ctx,
                             legion_index_space_t handle);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  legion_index_space_attach_name(legion_runtime_t runtime,
                                 legion_index_space_t handle,
                                 const char *name,
                                 bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  legion_index_space_retrieve_name(legion_runtime_t runtime,
                                   legion_index_space_t handle,
                                   const char **result);

  // -----------------------------------------------------------------------
  // Index Partition Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Coloring, bool, int)
   */
  legion_index_partition_t
  legion_index_partition_create_coloring(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_coloring_t coloring,
    bool disjoint,
    int part_color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
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
    int part_color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Domain, PointColoring, PartitionKind, int)
   */
  legion_index_partition_t
  legion_index_partition_create_point_coloring(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_domain_t color_space,
    legion_point_coloring_t coloring,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Domain, DomainPointColoring, PartitionKind, int)
   */
  legion_index_partition_t
  legion_index_partition_create_domain_point_coloring(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_domain_t color_space,
    legion_domain_point_coloring_t coloring,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition(
   *        Context, IndexSpace, Domain, MultiDomainPointColoring, PartitionKind, int)
   */
  legion_index_partition_t
  legion_index_partition_create_multi_domain_point_coloring(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_domain_t color_space,
    legion_multi_domain_point_coloring_t coloring,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition<T>(
   *        Context, IndexSpace, const T&, int)
   */
  legion_index_partition_t
  legion_index_partition_create_blockify_1d(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_blockify_1d_t blockify,
    int part_color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition<T>(
   *        Context, IndexSpace, const T&, int)
   */
  legion_index_partition_t
  legion_index_partition_create_blockify_2d(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_blockify_2d_t blockify,
    int part_color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_index_partition<T>(
   *        Context, IndexSpace, const T&, int)
   */
  legion_index_partition_t
  legion_index_partition_create_blockify_3d(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_blockify_3d_t blockify,
    int part_color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_equal_partition()
   */
  legion_index_partition_t
  legion_index_partition_create_equal(legion_runtime_t runtime,
                                      legion_context_t ctx,
                                      legion_index_space_t parent,
                                      legion_index_space_t color_space,
                                      size_t granularity,
                                      int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_union()
   */
  legion_index_partition_t
  legion_index_partition_create_by_union(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_index_partition_t handle1,
    legion_index_partition_t handle2,
    legion_index_space_t color_space,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_intersection()
   */
  legion_index_partition_t
  legion_index_partition_create_by_intersection(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_index_partition_t handle1,
    legion_index_partition_t handle2,
    legion_index_space_t color_space,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_difference()
   */
  legion_index_partition_t
  legion_index_partition_create_by_difference(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_index_partition_t handle1,
    legion_index_partition_t handle2,
    legion_index_space_t color_space,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_field()
   */
  legion_index_partition_t
  legion_index_partition_create_by_field(legion_runtime_t runtime,
                                         legion_context_t ctx,
                                         legion_logical_region_t handle,
                                         legion_logical_region_t parent,
                                         legion_field_id_t fid,
                                         legion_index_space_t color_space,
                                         int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_image()
   */
  legion_index_partition_t
  legion_index_partition_create_by_image(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t handle,
    legion_logical_partition_t projection,
    legion_logical_region_t parent,
    legion_field_id_t fid,
    legion_index_space_t color_space,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_preimage()
   */
  legion_index_partition_t
  legion_index_partition_create_by_preimage(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_partition_t projection,
    legion_logical_region_t handle,
    legion_logical_region_t parent,
    legion_field_id_t fid,
    legion_index_space_t color_space,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_image_range()
   */
  legion_index_partition_t
  legion_index_partition_create_by_image_range(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t handle,
    legion_logical_partition_t projection,
    legion_logical_region_t parent,
    legion_field_id_t fid,
    legion_index_space_t color_space,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_preimage()
   */
  legion_index_partition_t
  legion_index_partition_create_by_preimage_range(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_partition_t projection,
    legion_logical_region_t handle,
    legion_logical_region_t parent,
    legion_field_id_t fid,
    legion_index_space_t color_space,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_partition_by_restriction()
   */
  legion_index_partition_t
  legion_index_partition_create_by_restriction(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_index_space_t parent,
    legion_index_space_t color_space,
    legion_domain_transform_t transform,
    legion_domain_t extent,
    legion_partition_kind_t part_kind /* = COMPUTE_KIND */,
    int color /* = AUTO_GENERATE_ID */);

  /**
   * @see Legion::Runtime::is_index_partition_disjoint()
   */
  bool
  legion_index_partition_is_disjoint(legion_runtime_t runtime,
                                     legion_index_partition_t handle);

  /**
   * @see Legion::Runtime::is_index_partition_complete()
   */
  bool
  legion_index_partition_is_complete(legion_runtime_t runtime,
                                     legion_index_partition_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_index_subspace()
   */
  legion_index_space_t
  legion_index_partition_get_index_subspace(legion_runtime_t runtime,
                                            legion_index_partition_t handle,
                                            legion_color_t color);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_index_subspace()
   */
  legion_index_space_t
  legion_index_partition_get_index_subspace_domain_point(
    legion_runtime_t runtime,
    legion_index_partition_t handle,
    legion_domain_point_t color);

  /**
   * @see Legion::Runtime::has_index_subspace()
   */
  bool
  legion_index_partition_has_index_subspace_domain_point(
    legion_runtime_t runtime,
    legion_index_partition_t handle,
    legion_domain_point_t color);

  /**
   * @see Legion::Runtime::get_index_partition_color_space_name()
   */
  legion_index_space_t
  legion_index_partition_get_color_space(legion_runtime_t runtime,
                                         legion_index_partition_t handle);

  /**
   * @see Legion::Runtime::get_index_partition_color()
   */
  legion_color_t
  legion_index_partition_get_color(legion_runtime_t runtime,
                                   legion_index_partition_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_parent_index_space()
   */
  legion_index_space_t
  legion_index_partition_get_parent_index_space(legion_runtime_t runtime,
                                                legion_index_partition_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_index_space()
   */
  void
  legion_index_partition_destroy(legion_runtime_t runtime,
                                legion_context_t ctx,
                                 legion_index_partition_t handle);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  legion_index_partition_attach_name(legion_runtime_t runtime,
                                     legion_index_partition_t handle,
                                     const char *name,
                                     bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  legion_index_partition_retrieve_name(legion_runtime_t runtime,
                                       legion_index_partition_t handle,
                                       const char **result);

  // -----------------------------------------------------------------------
  // Field Space Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_field_space()
   */
  legion_field_space_t
  legion_field_space_create(legion_runtime_t runtime,
                            legion_context_t ctx);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_field_space()
   */
  void
  legion_field_space_destroy(legion_runtime_t runtime,
                             legion_context_t ctx,
                             legion_field_space_t handle);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  legion_field_space_attach_name(legion_runtime_t runtime,
                                 legion_field_space_t handle,
                                 const char *name,
                                 bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  legion_field_space_retrieve_name(legion_runtime_t runtime,
                                   legion_field_space_t handle,
                                   const char **result);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  legion_field_id_attach_name(legion_runtime_t runtime,
                              legion_field_space_t handle,
                              legion_field_id_t id,
                              const char *name,
                              bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  legion_field_id_retrieve_name(legion_runtime_t runtime,
                                legion_field_space_t handle,
                                legion_field_id_t id,
                                const char **result);

  // -----------------------------------------------------------------------
  // Logical Region Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_logical_region()
   */
  legion_logical_region_t
  legion_logical_region_create(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_index_space_t index,
                               legion_field_space_t fields,
                               bool task_local);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_logical_region()
   */
  void
  legion_logical_region_destroy(legion_runtime_t runtime,
                                legion_context_t ctx,
                                legion_logical_region_t handle);

  /**
   * @see Legion::Runtime::get_logical_region_color()
   */
  legion_color_t
  legion_logical_region_get_color(legion_runtime_t runtime,
                                  legion_logical_region_t handle);

  /**
   * @see Legion::Runtime::get_logical_region_color_point()
   */
  legion_domain_point_t
  legion_logical_region_get_color_domain_point(legion_runtime_t runtime_,
                                               legion_logical_region_t handle_);

  /**
   * @see Legion::Runtime::has_parent_logical_partition()
   */
  bool
  legion_logical_region_has_parent_logical_partition(
    legion_runtime_t runtime,
    legion_logical_region_t handle);

  /**
   * @see Legion::Runtime::get_parent_logical_partition()
   */
  legion_logical_partition_t
  legion_logical_region_get_parent_logical_partition(
    legion_runtime_t runtime,
    legion_logical_region_t handle);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  legion_logical_region_attach_name(legion_runtime_t runtime,
                                    legion_logical_region_t handle,
                                    const char *name,
                                    bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  legion_logical_region_retrieve_name(legion_runtime_t runtime,
                                      legion_logical_region_t handle,
                                      const char **result);

  // -----------------------------------------------------------------------
  // Logical Region Tree Traversal Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::get_logical_partition()
   */
  legion_logical_partition_t
  legion_logical_partition_create(legion_runtime_t runtime,
                                  legion_context_t ctx,
                                  legion_logical_region_t parent,
                                  legion_index_partition_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::get_logical_partition_by_tree()
   */
  legion_logical_partition_t
  legion_logical_partition_create_by_tree(legion_runtime_t runtime,
                                          legion_context_t ctx,
                                          legion_index_partition_t handle,
                                          legion_field_space_t fspace,
                                          legion_region_tree_id_t tid);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_logical_partition()
   */
  void
  legion_logical_partition_destroy(legion_runtime_t runtime,
                                   legion_context_t ctx,
                                   legion_logical_partition_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_logical_subregion()
   */
  legion_logical_region_t
  legion_logical_partition_get_logical_subregion(
    legion_runtime_t runtime,
    legion_logical_partition_t parent,
    legion_index_space_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_logical_subregion_by_color()
   */
  legion_logical_region_t
  legion_logical_partition_get_logical_subregion_by_color(
    legion_runtime_t runtime,
    legion_logical_partition_t parent,
    legion_color_t c);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_logical_subregion_by_color()
   */
  legion_logical_region_t
  legion_logical_partition_get_logical_subregion_by_color_domain_point(
    legion_runtime_t runtime,
    legion_logical_partition_t parent,
    legion_domain_point_t c);

  /**
   * @see Legion::Runtime::has_logical_subregion_by_color()
   */
  bool
  legion_logical_partition_has_logical_subregion_by_color_domain_point(
    legion_runtime_t runtime,
    legion_logical_partition_t parent,
    legion_domain_point_t c);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::get_logical_subregion_by_tree()
   */
  legion_logical_region_t
  legion_logical_partition_get_logical_subregion_by_tree(
    legion_runtime_t runtime,
    legion_index_space_t handle,
    legion_field_space_t fspace,
    legion_region_tree_id_t tid);

  /**
   * @see Legion::Runtime::get_parent_logical_region()
   */
  legion_logical_region_t
  legion_logical_partition_get_parent_logical_region(
    legion_runtime_t runtime,
    legion_logical_partition_t handle);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  legion_logical_partition_attach_name(legion_runtime_t runtime,
                                       legion_logical_partition_t handle,
                                       const char *name,
                                       bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  legion_logical_partition_retrieve_name(legion_runtime_t runtime,
                                         legion_logical_partition_t handle,
                                         const char **result);

  // -----------------------------------------------------------------------
  // Region Requirement Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::RegionRequirement::region
   */
  legion_logical_region_t
  legion_region_requirement_get_region(legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::parent
   */
  legion_logical_region_t
  legion_region_requirement_get_parent(legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::partition
   */
  legion_logical_partition_t
  legion_region_requirement_get_partition(legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::privilege_fields
   */
  unsigned
  legion_region_requirement_get_privilege_fields_size(
      legion_region_requirement_t handle);

  /**
   * @param fields Caller should give a buffer of the size fields_size
   *
   * @param fields_size the size of the buffer fields
   *
   * @return returns privilege fields in the region requirement.
   *         The return might be truncated if the buffer size is
   *         smaller than the number of privilege fields.
   *
   * @see Legion::RegionRequirement::privilege_fields
   */
  void
  legion_region_requirement_get_privilege_fields(
      legion_region_requirement_t handle,
      legion_field_id_t* fields,
      unsigned fields_size);

  /**
   * @return returns the i-th privilege field in the region requirement.
   *         note that this function takes O(n) time due to the underlying
   *         data structure does not provide an indexing operation.
   *
   * @see Legion::RegionRequirement::privilege_fields
   */
  legion_field_id_t
  legion_region_requirement_get_privilege_field(
      legion_region_requirement_t handle,
      unsigned idx);

  /**
   * @see Legion::RegionRequirement::instance_fields
   */
  unsigned
  legion_region_requirement_get_instance_fields_size(
      legion_region_requirement_t handle);

  /**
   * @param fields Caller should give a buffer of the size fields_size
   *
   * @param fields_size the size of the buffer fields
   *
   * @return returns instance fields in the region requirement.
   *         The return might be truncated if the buffer size is
   *         smaller than the number of instance fields.
   *
   * @see Legion::RegionRequirement::instance_fields
   */
  void
  legion_region_requirement_get_instance_fields(
      legion_region_requirement_t handle,
      legion_field_id_t* fields,
      unsigned fields_size);

  /**
   * @return returns the i-th instance field in the region requirement.
   *
   * @see Legion::RegionRequirement::instance_fields
   */
  legion_field_id_t
  legion_region_requirement_get_instance_field(
      legion_region_requirement_t handle,
      unsigned idx);

  /**
   * @see Legion::RegionRequirement::privilege
   */
  legion_privilege_mode_t
  legion_region_requirement_get_privilege(legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::prop
   */
  legion_coherence_property_t
  legion_region_requirement_get_prop(legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::redop
   */
  legion_reduction_op_id_t
  legion_region_requirement_get_redop(legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::tag
   */
  legion_mapping_tag_id_t
  legion_region_requirement_get_tag(legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::handle_type
   */
  legion_handle_type_t
  legion_region_requirement_get_handle_type(legion_region_requirement_t handle);

  /**
   * @see Legion::RegionRequirement::projection
   */
  legion_projection_id_t
  legion_region_requirement_get_projection(legion_region_requirement_t handle);

  // -----------------------------------------------------------------------
  // Allocator and Argument Map Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_field_allocator()
   */
  legion_field_allocator_t
  legion_field_allocator_create(legion_runtime_t runtime,
                                legion_context_t ctx,
                                legion_field_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::FieldAllocator::~FieldAllocator()
   */
  void
  legion_field_allocator_destroy(legion_field_allocator_t handle);

  /**
   * @see Legion::FieldAllocator::allocate_field()
   */
  legion_field_id_t
  legion_field_allocator_allocate_field(
    legion_field_allocator_t allocator,
    size_t field_size,
    legion_field_id_t desired_fieldid /* = AUTO_GENERATE_ID */);

  /**
   * @see Legion::FieldAllocator::free_field()
   */
  void
  legion_field_allocator_free_field(legion_field_allocator_t allocator,
                                    legion_field_id_t fid);

  /**
   * @see Legion::FieldAllocator::allocate_local_field()
   */
  legion_field_id_t
  legion_field_allocator_allocate_local_field(
    legion_field_allocator_t allocator,
    size_t field_size,
    legion_field_id_t desired_fieldid /* = AUTO_GENERATE_ID */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::ArgumentMap::ArgumentMap()
   */
  legion_argument_map_t
  legion_argument_map_create();

  /**
   * @see Legion::ArgumentMap::set_point()
   */
  void
  legion_argument_map_set_point(legion_argument_map_t map,
                                legion_domain_point_t dp,
                                legion_task_argument_t arg,
                                bool replace /* = true */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::ArgumentMap::~ArgumentMap()
   */
  void
  legion_argument_map_destroy(legion_argument_map_t handle);

  // -----------------------------------------------------------------------
  // Predicate Operations
  // -----------------------------------------------------------------------

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Predicate::~Predicate()
   */
  void
  legion_predicate_destroy(legion_predicate_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Predicate::TRUE_PRED
   */
  const legion_predicate_t
  legion_predicate_true(void);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Predicate::FALSE_PRED
   */
  const legion_predicate_t
  legion_predicate_false(void);

  // -----------------------------------------------------------------------
  // Phase Barrier Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_phase_barrier()
   */
  legion_phase_barrier_t
  legion_phase_barrier_create(legion_runtime_t runtime,
                              legion_context_t ctx,
                              unsigned arrivals);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_phase_barrier()
   */
  void
  legion_phase_barrier_destroy(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_phase_barrier_t handle);

  /**
   * @see Legion::PhaseBarrier::alter_arrival_count()
   */
  legion_phase_barrier_t
  legion_phase_barrier_alter_arrival_count(legion_runtime_t runtime,
                                           legion_context_t ctx,
                                           legion_phase_barrier_t handle,
                                           int delta);

  /**
   * @see Legion::PhaseBarrier::arrive()
   */
  void
  legion_phase_barrier_arrive(legion_runtime_t runtime,
                              legion_context_t ctx,
                              legion_phase_barrier_t handle,
                              unsigned count /* = 1 */);

  /**
   * @see Legion::PhaseBarrier::wait()
   */
  void
  legion_phase_barrier_wait(legion_runtime_t runtime,
                            legion_context_t ctx,
                            legion_phase_barrier_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::advance_phase_barrier()
   */
  legion_phase_barrier_t
  legion_phase_barrier_advance(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_phase_barrier_t handle);

  // -----------------------------------------------------------------------
  // Dynamic Collective Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::create_dynamic_collective()
   */
  legion_dynamic_collective_t
  legion_dynamic_collective_create(legion_runtime_t runtime,
                                   legion_context_t ctx,
                                   unsigned arrivals,
                                   legion_reduction_op_id_t redop,
                                   const void *init_value,
                                   size_t init_size);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Runtime::destroy_dynamic_collective()
   */
  void
  legion_dynamic_collective_destroy(legion_runtime_t runtime,
                                    legion_context_t ctx,
                                    legion_dynamic_collective_t handle);

  /**
   * @see Legion::DynamicCollective::alter_arrival_count()
   */
  legion_dynamic_collective_t
  legion_dynamic_collective_alter_arrival_count(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_dynamic_collective_t handle,
    int delta);

  /**
   * @see Legion::Runtime::arrive_dynamic_collective()
   */
  void
  legion_dynamic_collective_arrive(legion_runtime_t runtime,
                                   legion_context_t ctx,
                                   legion_dynamic_collective_t handle,
                                   const void *buffer,
                                   size_t size,
                                   unsigned count /* = 1 */);

  /**
   * @see Legion::Runtime::defer_dynamic_collective_arrival()
   */
  void
  legion_dynamic_collective_defer_arrival(legion_runtime_t runtime,
                                          legion_context_t ctx,
                                          legion_dynamic_collective_t handle,
                                          legion_future_t f,
                                          unsigned count /* = 1 */);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::get_dynamic_collective_result()
   */
  legion_future_t
  legion_dynamic_collective_get_result(legion_runtime_t runtime,
                                       legion_context_t ctx,
                                       legion_dynamic_collective_t handle);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Runtime::advance_dynamic_collective()
   */
  legion_dynamic_collective_t
  legion_dynamic_collective_advance(legion_runtime_t runtime,
                                    legion_context_t ctx,
                                    legion_dynamic_collective_t handle);

  // -----------------------------------------------------------------------
  // Future Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Future::from_untyped_pointer()
   */
  legion_future_t
  legion_future_from_untyped_pointer(legion_runtime_t runtime,
                                     const void *buffer,
                                     size_t size);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Future::Future()
   */
  legion_future_t
  legion_future_copy(legion_future_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Future::~Future()
   */
  void
  legion_future_destroy(legion_future_t handle);

  /**
   * @see Legion::Future::get_void_result()
   */
  void
  legion_future_get_void_result(legion_future_t handle);

  /**
   * @see Legion::Future::is_empty()
   */
  bool
  legion_future_is_empty(legion_future_t handle,
                         bool block /* = false */);

  /**
   * @see Legion::Future::is_ready()
   */
  bool
  legion_future_is_ready(legion_future_t handle);

  /**
   * @see Legion::Future::get_untyped_pointer()
   */
  const void *
  legion_future_get_untyped_pointer(legion_future_t handle);

  /**
   * @see Legion::Future::get_untyped_size()
   */
  size_t
  legion_future_get_untyped_size(legion_future_t handle);

  // -----------------------------------------------------------------------
  // Future Map Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::FutureMap::FutureMap()
   */
  legion_future_map_t
  legion_future_map_copy(legion_future_map_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::FutureMap::~FutureMap()
   */
  void
  legion_future_map_destroy(legion_future_map_t handle);

  /**
   * @see Legion::FutureMap::wait_all_results()
   */
  void
  legion_future_map_wait_all_results(legion_future_map_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Future::get_future()
   */
  legion_future_t
  legion_future_map_get_future(legion_future_map_t handle,
                               legion_domain_point_t point);

  // -----------------------------------------------------------------------
  // Task Launch Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::TaskLauncher::TaskLauncher()
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
   * @see Legion::TaskLauncher::~TaskLauncher()
   */
  void
  legion_task_launcher_destroy(legion_task_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_task()
   */
  legion_future_t
  legion_task_launcher_execute(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_task_launcher_t launcher);

  /**
   * @see Legion::TaskLauncher::add_region_requirement()
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
   * @see Legion::TaskLauncher::add_region_requirement()
   */
  unsigned
  legion_task_launcher_add_region_requirement_logical_region_reduction(
    legion_task_launcher_t launcher,
    legion_logical_region_t handle,
    legion_reduction_op_id_t redop,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::TaskLauncher::add_field()
   */
  void
  legion_task_launcher_add_field(legion_task_launcher_t launcher,
                                 unsigned idx,
                                 legion_field_id_t fid,
                                 bool inst /* = true */);

  /**
   * @see Legion::RegionRequirement::add_flags()
   */
  void
  legion_task_launcher_add_flags(legion_task_launcher_t launcher,
                                 unsigned idx,
                                 enum legion_region_flags_t flags);

  /**
   * @see Legion::RegionRequirement::flags
   */
  void
  legion_task_launcher_intersect_flags(legion_task_launcher_t launcher,
                                       unsigned idx,
                                       enum legion_region_flags_t flags);

  /**
   * @see Legion::TaskLauncher::add_index_requirement()
   */
  unsigned
  legion_task_launcher_add_index_requirement(
    legion_task_launcher_t launcher,
    legion_index_space_t handle,
    legion_allocate_mode_t priv,
    legion_index_space_t parent,
    bool verified /* = false*/);

  /**
   * @see Legion::TaskLauncher::add_future()
   */
  void
  legion_task_launcher_add_future(legion_task_launcher_t launcher,
                                  legion_future_t future);

  /**
   * @see Legion::TaskLauncher::add_wait_barrier()
   */
  void
  legion_task_launcher_add_wait_barrier(legion_task_launcher_t launcher,
                                        legion_phase_barrier_t bar);

  /**
   * @see Legion::TaskLauncher::add_arrival_barrier()
   */
  void
  legion_task_launcher_add_arrival_barrier(legion_task_launcher_t launcher,
                                           legion_phase_barrier_t bar);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::IndexTaskLauncher::IndexTaskLauncher()
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
   * @see Legion::IndexTaskLauncher::~IndexTaskLauncher()
   */
  void
  legion_index_launcher_destroy(legion_index_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &)
   */
  legion_future_map_t
  legion_index_launcher_execute(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_index_launcher_t launcher);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, ReductionOpID)
   */
  legion_future_t
  legion_index_launcher_execute_reduction(legion_runtime_t runtime,
                                          legion_context_t ctx,
                                          legion_index_launcher_t launcher,
                                          legion_reduction_op_id_t redop);

  /**
   * @see Legion::IndexTaskLauncher::add_region_requirement()
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
   * @see Legion::IndexTaskLauncher::add_region_requirement()
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
   * @see Legion::IndexTaskLauncher::add_region_requirement()
   */
  unsigned
  legion_index_launcher_add_region_requirement_logical_region_reduction(
    legion_index_launcher_t launcher,
    legion_logical_region_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_reduction_op_id_t redop,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::add_region_requirement()
   */
  unsigned
  legion_index_launcher_add_region_requirement_logical_partition_reduction(
    legion_index_launcher_t launcher,
    legion_logical_partition_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_reduction_op_id_t redop,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexLaunchxer::add_field()
   */
  void
  legion_index_launcher_add_field(legion_index_launcher_t launcher,
                                 unsigned idx,
                                 legion_field_id_t fid,
                                 bool inst /* = true */);

  /**
   * @see Legion::RegionRequirement::add_flags()
   */
  void
  legion_index_launcher_add_flags(legion_index_launcher_t launcher,
                                  unsigned idx,
                                  enum legion_region_flags_t flags);

  /**
   * @see Legion::RegionRequirement::flags
   */
  void
  legion_index_launcher_intersect_flags(legion_index_launcher_t launcher,
                                        unsigned idx,
                                        enum legion_region_flags_t flags);

  /**
   * @see Legion::IndexTaskLauncher::add_index_requirement()
   */
  unsigned
  legion_index_launcher_add_index_requirement(
    legion_index_launcher_t launcher,
    legion_index_space_t handle,
    legion_allocate_mode_t priv,
    legion_index_space_t parent,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexTaskLauncher::add_future()
   */
  void
  legion_index_launcher_add_future(legion_index_launcher_t launcher,
                                   legion_future_t future);

  /**
   * @see Legion::IndexTaskLauncher::add_wait_barrier()
   */
  void
  legion_index_launcher_add_wait_barrier(legion_index_launcher_t launcher,
                                         legion_phase_barrier_t bar);

  /**
   * @see Legion::IndexTaskLauncher::add_arrival_barrier()
   */
  void
  legion_index_launcher_add_arrival_barrier(legion_index_launcher_t launcher,
                                            legion_phase_barrier_t bar);

  // -----------------------------------------------------------------------
  // Inline Mapping Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::InlineLauncher::InlineLauncher()
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
   * @see Legion::InlineLauncher::~InlineLauncher()
   */
  void
  legion_inline_launcher_destroy(legion_inline_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::map_region()
   */
  legion_physical_region_t
  legion_inline_launcher_execute(legion_runtime_t runtime,
                                 legion_context_t ctx,
                                 legion_inline_launcher_t launcher);

  /**
   * @see Legion::InlineLauncher::add_field()
   */
  void
  legion_inline_launcher_add_field(legion_inline_launcher_t launcher,
                                   legion_field_id_t fid,
                                   bool inst /* = true */);

  /**
   * @see Legion::Runtime::remap_region()
   */
  void
  legion_runtime_remap_region(legion_runtime_t runtime,
                              legion_context_t ctx,
                              legion_physical_region_t region);

  /**
   * @see Legion::Runtime::unmap_region()
   */
  void
  legion_runtime_unmap_region(legion_runtime_t runtime,
                              legion_context_t ctx,
                              legion_physical_region_t region);

  /**
   * @see Legion::Runtime::unmap_all_regions()
   */
  void
  legion_runtime_unmap_all_regions(legion_runtime_t runtime,
                                   legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Fill Field Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::fill_field()
   */
  void
  legion_runtime_fill_field(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_logical_region_t handle,
    legion_logical_region_t parent,
    legion_field_id_t fid,
    const void *value,
    size_t value_size,
    legion_predicate_t pred /* = legion_predicate_true() */);

  /**
   * @see Legion::Runtime::fill_field()
   */
  void
  legion_runtime_fill_field_future(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_logical_region_t handle,
    legion_logical_region_t parent,
    legion_field_id_t fid,
    legion_future_t f,
    legion_predicate_t pred /* = legion_predicate_true() */);

  // -----------------------------------------------------------------------
  // File Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   */
  legion_field_map_t
  legion_field_map_create();

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   */
  void
  legion_field_map_destroy(legion_field_map_t handle);

  void
  legion_field_map_insert(legion_field_map_t handle,
                          legion_field_id_t key,
                          const char *value);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::attach_hdf5()
   */
  legion_physical_region_t
  legion_runtime_attach_hdf5(
    legion_runtime_t runtime,
    legion_context_t ctx,
    const char *filename,
    legion_logical_region_t handle,
    legion_logical_region_t parent,
    legion_field_map_t field_map,
    legion_file_mode_t mode);

  /**
   * @see Legion::Runtime::detach_hdf5()
   */
  void
  legion_runtime_detach_hdf5(
    legion_runtime_t runtime,
    legion_context_t ctx,
    legion_physical_region_t region);

  // -----------------------------------------------------------------------
  // Copy Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::CopyLauncher::CopyLauncher()
   */
  legion_copy_launcher_t
  legion_copy_launcher_create(
    legion_predicate_t pred /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::CopyLauncher::~CopyLauncher()
   */
  void
  legion_copy_launcher_destroy(legion_copy_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::issue_copy_operation()
   */
  void
  legion_copy_launcher_execute(legion_runtime_t runtime,
                               legion_context_t ctx,
                               legion_copy_launcher_t launcher);

  /**
   * @see Legion::CopyLauncher::add_copy_requirements()
   */
  unsigned
  legion_copy_launcher_add_src_region_requirement_logical_region(
    legion_copy_launcher_t launcher,
    legion_logical_region_t handle,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::CopyLauncher::add_copy_requirements()
   */
  unsigned
  legion_copy_launcher_add_dst_region_requirement_logical_region(
    legion_copy_launcher_t launcher,
    legion_logical_region_t handle,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::CopyLauncher::add_region_requirement()
   */
  unsigned
  legion_copy_launcher_add_dst_region_requirement_logical_region_reduction(
    legion_copy_launcher_t launcher,
    legion_logical_region_t handle,
    legion_reduction_op_id_t redop,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::CopyLauncher::add_src_field()
   */
  void
  legion_copy_launcher_add_src_field(legion_copy_launcher_t launcher,
                                     unsigned idx,
                                     legion_field_id_t fid,
                                     bool inst /* = true */);

  /**
   * @see Legion::CopyLauncher::add_dst_field()
   */
  void
  legion_copy_launcher_add_dst_field(legion_copy_launcher_t launcher,
                                     unsigned idx,
                                     legion_field_id_t fid,
                                     bool inst /* = true */);

  /**
   * @see Legion::CopyLauncher::add_wait_barrier()
   */
  void
  legion_copy_launcher_add_wait_barrier(legion_copy_launcher_t launcher,
                                        legion_phase_barrier_t bar);

  /**
   * @see Legion::CopyLauncher::add_arrival_barrier()
   */
  void
  legion_copy_launcher_add_arrival_barrier(legion_copy_launcher_t launcher,
                                           legion_phase_barrier_t bar);

  // -----------------------------------------------------------------------
  // Index Copy Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::IndexCopyLauncher::IndexCopyLauncher()
   */
  legion_index_copy_launcher_t
  legion_index_copy_launcher_create(
    legion_domain_t domain,
    legion_predicate_t pred /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::IndexCopyLauncher::~IndexCopyLauncher()
   */
  void
  legion_index_copy_launcher_destroy(legion_index_copy_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::issue_index_copy_operation()
   */
  void
  legion_index_copy_launcher_execute(legion_runtime_t runtime,
                                     legion_context_t ctx,
                                     legion_index_copy_launcher_t launcher);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  legion_index_copy_launcher_add_src_region_requirement_logical_region(
    legion_index_copy_launcher_t launcher,
    legion_logical_region_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  legion_index_copy_launcher_add_dst_region_requirement_logical_region(
    legion_index_copy_launcher_t launcher,
    legion_logical_region_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  legion_index_copy_launcher_add_src_region_requirement_logical_partition(
    legion_index_copy_launcher_t launcher,
    legion_logical_partition_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  legion_index_copy_launcher_add_dst_region_requirement_logical_partition(
    legion_index_copy_launcher_t launcher,
    legion_logical_partition_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_privilege_mode_t priv,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  legion_index_copy_launcher_add_dst_region_requirement_logical_region_reduction(
    legion_index_copy_launcher_t launcher,
    legion_logical_region_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_reduction_op_id_t redop,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_copy_requirements()
   */
  unsigned
  legion_index_copy_launcher_add_dst_region_requirement_logical_partition_reduction(
    legion_index_copy_launcher_t launcher,
    legion_logical_partition_t handle,
    legion_projection_id_t proj /* = 0 */,
    legion_reduction_op_id_t redop,
    legion_coherence_property_t prop,
    legion_logical_region_t parent,
    legion_mapping_tag_id_t tag /* = 0 */,
    bool verified /* = false*/);

  /**
   * @see Legion::IndexCopyLauncher::add_src_field()
   */
  void
  legion_index_copy_launcher_add_src_field(legion_index_copy_launcher_t launcher,
                                           unsigned idx,
                                           legion_field_id_t fid,
                                           bool inst /* = true */);

  /**
   * @see Legion::IndexCopyLauncher::add_dst_field()
   */
  void
  legion_index_copy_launcher_add_dst_field(legion_index_copy_launcher_t launcher,
                                           unsigned idx,
                                           legion_field_id_t fid,
                                           bool inst /* = true */);

  /**
   * @see Legion::IndexCopyLauncher::add_wait_barrier()
   */
  void
  legion_index_copy_launcher_add_wait_barrier(legion_index_copy_launcher_t launcher,
                                              legion_phase_barrier_t bar);

  /**
   * @see Legion::IndexCopyLauncher::add_arrival_barrier()
   */
  void
  legion_index_copy_launcher_add_arrival_barrier(legion_index_copy_launcher_t launcher,
                                                 legion_phase_barrier_t bar);

  // -----------------------------------------------------------------------
  // Acquire Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::AcquireLauncher::AcquireLauncher()
   */
  legion_acquire_launcher_t
  legion_acquire_launcher_create(
    legion_logical_region_t logical_region,
    legion_logical_region_t parent_region,
    legion_predicate_t pred /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::AcquireLauncher::~AcquireLauncher()
   */
  void
  legion_acquire_launcher_destroy(legion_acquire_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::issue_acquire()
   */
  void
  legion_acquire_launcher_execute(legion_runtime_t runtime,
                                  legion_context_t ctx,
                                  legion_acquire_launcher_t launcher);

  /**
   * @see Legion::AcquireLauncher::add_field()
   */
  void
  legion_acquire_launcher_add_field(legion_acquire_launcher_t launcher,
                                    legion_field_id_t fid);

  /**
   * @see Legion::AcquireLauncher::add_wait_barrier()
   */
  void
  legion_acquire_launcher_add_wait_barrier(legion_acquire_launcher_t launcher,
                                           legion_phase_barrier_t bar);

  /**
   * @see Legion::AcquireLauncher::add_arrival_barrier()
   */
  void
  legion_acquire_launcher_add_arrival_barrier(
    legion_acquire_launcher_t launcher,
    legion_phase_barrier_t bar);

  // -----------------------------------------------------------------------
  // Release Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::ReleaseLauncher::ReleaseLauncher()
   */
  legion_release_launcher_t
  legion_release_launcher_create(
    legion_logical_region_t logical_region,
    legion_logical_region_t parent_region,
    legion_predicate_t pred /* = legion_predicate_true() */,
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::ReleaseLauncher::~ReleaseLauncher()
   */
  void
  legion_release_launcher_destroy(legion_release_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::issue_release()
   */
  void
  legion_release_launcher_execute(legion_runtime_t runtime,
                                  legion_context_t ctx,
                                  legion_release_launcher_t launcher);

  /**
   * @see Legion::ReleaseLauncher::add_field()
   */
  void
  legion_release_launcher_add_field(legion_release_launcher_t launcher,
                                    legion_field_id_t fid);

  /**
   * @see Legion::ReleaseLauncher::add_wait_barrier()
   */
  void
  legion_release_launcher_add_wait_barrier(legion_release_launcher_t launcher,
                                           legion_phase_barrier_t bar);

  /**
   * @see Legion::ReleaseLauncher::add_arrival_barrier()
   */
  void
  legion_release_launcher_add_arrival_barrier(
    legion_release_launcher_t launcher,
    legion_phase_barrier_t bar);

  // -----------------------------------------------------------------------
  // Attach/Detach Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::AttachLauncher::AttachLauncher()
   */
  legion_attach_launcher_t
  legion_attach_launcher_create(
    legion_logical_region_t logical_region,
    legion_logical_region_t parent_region,
    legion_external_resource_t resource);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::AttachLauncher::~AttachLauncher()
   */
  void
  legion_attach_launcher_destroy(legion_attach_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::attach_external_resource()
   */
  legion_physical_region_t
  legion_attach_launcher_execute(legion_runtime_t runtime,
                                 legion_context_t ctx,
                                 legion_attach_launcher_t launcher);

  /**
   * @see Legion::AttachLauncher::attach_array_soa()
   */
  void
  legion_attach_launcher_add_cpu_soa_field(legion_attach_launcher_t launcher,
                                           legion_field_id_t fid,
                                           void *base_ptr,
                                           bool column_major);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::detach_external_resource()
   */
  legion_future_t
  legion_detach_external_resource(legion_runtime_t runtime,
                                  legion_context_t ctx,
                                  legion_physical_region_t handle);

  // -----------------------------------------------------------------------
  // Must Epoch Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::MustEpochLauncher::MustEpochLauncher()
   */
  legion_must_epoch_launcher_t
  legion_must_epoch_launcher_create(
    legion_mapper_id_t id /* = 0 */,
    legion_mapping_tag_id_t launcher_tag /* = 0 */);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::MustEpochLauncher::~MustEpochLauncher()
   */
  void
  legion_must_epoch_launcher_destroy(legion_must_epoch_launcher_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::execute_must_epoch()
   */
  legion_future_map_t
  legion_must_epoch_launcher_execute(legion_runtime_t runtime,
                                     legion_context_t ctx,
                                     legion_must_epoch_launcher_t launcher);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Must_EpochLauncher::add_single_task()
   */
  void
  legion_must_epoch_launcher_add_single_task(
    legion_must_epoch_launcher_t launcher,
    legion_domain_point_t point,
    legion_task_launcher_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Must_EpochLauncher::add_index_task()
   */
  void
  legion_must_epoch_launcher_add_index_task(
    legion_must_epoch_launcher_t launcher,
    legion_index_launcher_t handle);

  // -----------------------------------------------------------------------
  // Fence Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::issue_mapping_fence()
   */
  void
  legion_runtime_issue_mapping_fence(legion_runtime_t runtime,
                                     legion_context_t ctx);

  /**
   * @see Legion::Runtime::issue_execution_fence()
   */
  void
  legion_runtime_issue_execution_fence(legion_runtime_t runtime,
                                       legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Tracing Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::begin_trace()
   */
  void
  legion_runtime_begin_trace(legion_runtime_t runtime,
                             legion_context_t ctx,
                             legion_trace_id_t tid);

  /**
   * @see Legion::Runtime::end_trace()
   */
  void
  legion_runtime_end_trace(legion_runtime_t runtime,
                           legion_context_t ctx,
                           legion_trace_id_t tid);

  // -----------------------------------------------------------------------
  // Tunable Variables
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::Runtime::select_tunable_value()
   */
  legion_future_t
  legion_runtime_select_tunable_value(legion_runtime_t runtime,
				      legion_context_t ctx,
				      legion_tunable_id_t tid,
				      legion_mapper_id_t mapper /* = 0 */,
				      legion_mapping_tag_id_t tag /* = 0 */);

  // -----------------------------------------------------------------------
  // Miscellaneous Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::get_runtime()
   */
  legion_runtime_t
  legion_runtime_get_runtime();

  /**
   * @see Legion::Runtime::get_executing_processor()
   */
  legion_processor_t
  legion_runtime_get_executing_processor(legion_runtime_t runtime,
                                         legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Physical Data Operations
  // -----------------------------------------------------------------------

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::PhysicalRegion::~PhysicalRegion()
   */
  void
  legion_physical_region_destroy(legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::is_mapped()
   */
  bool
  legion_physical_region_is_mapped(legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::wait_until_valid()
   */
  void
  legion_physical_region_wait_until_valid(legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::is_valid()
   */
  bool
  legion_physical_region_is_valid(legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::get_logical_region()
   */
  legion_logical_region_t
  legion_physical_region_get_logical_region(legion_physical_region_t handle);

  /**
   * @see Legion::PhysicalRegion::get_fields()
   */
  size_t
  legion_physical_region_get_field_count(legion_physical_region_t handle);
  legion_field_id_t
  legion_physical_region_get_field_id(legion_physical_region_t handle, size_t index);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::PhysicalRegion::get_field_accessor()
   */
  legion_accessor_array_1d_t
  legion_physical_region_get_field_accessor_array_1d(
    legion_physical_region_t handle,
    legion_field_id_t fid);
  
  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::PhysicalRegion::get_field_accessor()
   */
  legion_accessor_array_2d_t
  legion_physical_region_get_field_accessor_array_2d(
    legion_physical_region_t handle,
    legion_field_id_t fid);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::PhysicalRegion::get_field_accessor()
   */
  legion_accessor_array_3d_t
  legion_physical_region_get_field_accessor_array_3d(
    legion_physical_region_t handle,
    legion_field_id_t fid);

  void *
  legion_accessor_array_1d_raw_rect_ptr(legion_accessor_array_1d_t handle,
                                        legion_rect_1d_t rect,
                                        legion_rect_1d_t *subrect,
                                        legion_byte_offset_t *offsets);

  void *
  legion_accessor_array_2d_raw_rect_ptr(legion_accessor_array_2d_t handle,
                                        legion_rect_2d_t rect,
                                        legion_rect_2d_t *subrect,
                                        legion_byte_offset_t *offsets);

  void *
  legion_accessor_array_3d_raw_rect_ptr(legion_accessor_array_3d_t handle,
                                        legion_rect_3d_t rect,
                                        legion_rect_3d_t *subrect,
                                        legion_byte_offset_t *offsets);

  // Read
  void
  legion_accessor_array_1d_read(legion_accessor_array_1d_t handle,
                                legion_ptr_t ptr,
                                void *dst, size_t bytes);

  void
  legion_accessor_array_1d_read_point(legion_accessor_array_1d_t handle,
                                      legion_point_1d_t point,
                                      void *dst, size_t bytes);

  void
  legion_accessor_array_2d_read_point(legion_accessor_array_2d_t handle,
                                      legion_point_2d_t point,
                                      void *dst, size_t bytes);
  
  void
  legion_accessor_array_3d_read_point(legion_accessor_array_3d_t handle,
                                      legion_point_3d_t point,
                                      void *dst, size_t bytes);

  // Write
  void
  legion_accessor_array_1d_write(legion_accessor_array_1d_t handle,
                                 legion_ptr_t ptr,
                                 const void *src, size_t bytes);

  void
  legion_accessor_array_1d_write_point(legion_accessor_array_1d_t handle,
                                       legion_point_1d_t point,
                                       const void *src, size_t bytes);

  void
  legion_accessor_array_2d_write_point(legion_accessor_array_2d_t handle,
                                       legion_point_2d_t point,
                                       const void *src, size_t bytes);

  void
  legion_accessor_array_3d_write_point(legion_accessor_array_3d_t handle,
                                       legion_point_3d_t point,
                                       const void *src, size_t bytes);

  // Ref
  void *
  legion_accessor_array_1d_ref(legion_accessor_array_1d_t handle,
                               legion_ptr_t ptr);

  void *
  legion_accessor_array_1d_ref_point(legion_accessor_array_1d_t handle,
                                     legion_point_1d_t point);

  void *
  legion_accessor_array_2d_ref_point(legion_accessor_array_2d_t handle,
                                     legion_point_2d_t point);

  void *
  legion_accessor_array_3d_ref_point(legion_accessor_array_3d_t handle,
                                     legion_point_3d_t point);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   */
  void
  legion_accessor_array_1d_destroy(legion_accessor_array_1d_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   */
  void
  legion_accessor_array_2d_destroy(legion_accessor_array_2d_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   */
  void
  legion_accessor_array_3d_destroy(legion_accessor_array_3d_t handle);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Legion::IndexIterator::IndexIterator()
   */
  legion_index_iterator_t
  legion_index_iterator_create(legion_runtime_t runtime,
                               legion_context_t context,
                               legion_index_space_t handle);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::IndexIterator::~IndexIterator()
   */
  void
  legion_index_iterator_destroy(legion_index_iterator_t handle);

  /**
   * @see Legion::IndexIterator::has_next()
   */
  bool
  legion_index_iterator_has_next(legion_index_iterator_t handle);

  /**
   * @see Legion::IndexIterator::next()
   */
  legion_ptr_t
  legion_index_iterator_next(legion_index_iterator_t handle);

  /**
   * @see Legion::IndexIterator::next_span()
   */
  legion_ptr_t
  legion_index_iterator_next_span(legion_index_iterator_t handle,
                                  size_t *act_count,
                                  size_t req_count /* = -1 */);

  // -----------------------------------------------------------------------
  // Task Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Mappable::get_unique_id()
   */
  legion_unique_id_t
  legion_context_get_unique_id(legion_context_t ctx);

  /**
   * @see Legion::Mappable::get_unique_id()
   */
  legion_unique_id_t
  legion_task_get_unique_id(legion_task_t task);

  /**
   * @see Legion::Mappable::get_depth()
   */
  int
  legion_task_get_depth(legion_task_t task);

  /**
   * @see Legion::Mappable::tag
   */
  legion_mapping_tag_id_t
  legion_task_get_tag(legion_task_t task);

  /**
   * @see Legion::Runtime::attach_name()
   */
  void
  legion_task_id_attach_name(legion_runtime_t runtime,
                             legion_task_id_t task_id,
                             const char *name,
                             bool is_mutable /* = false */);

  /**
   * @see Legion::Runtime::retrieve_name()
   */
  void
  legion_task_id_retrieve_name(legion_runtime_t runtime,
                               legion_task_id_t task_id,
                               const char **result);

  /**
   * @see Legion::Task::args
   */
  void *
  legion_task_get_args(legion_task_t task);

  /**
   * @see Legion::Task::arglen
   */
  size_t
  legion_task_get_arglen(legion_task_t task);

  /**
   * @see Legion::Task::index_domain
   */
  legion_domain_t
  legion_task_get_index_domain(legion_task_t task);

  /**
   * @see Legion::Task::index_point
   */
  legion_domain_point_t
  legion_task_get_index_point(legion_task_t task);

  /**
   * @see Legion::Task::is_index_space
   */
  bool
  legion_task_get_is_index_space(legion_task_t task);

  /**
   * @see Legion::Task::local_args
   */
  void *
  legion_task_get_local_args(legion_task_t task);

  /**
   * @see Legion::Task::local_arglen
   */
  size_t
  legion_task_get_local_arglen(legion_task_t task);

  /**
   * @see Legion::Task::regions
   */
  unsigned
  legion_task_get_regions_size(legion_task_t task);

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Task::regions
   */
  legion_region_requirement_t
  legion_task_get_region(legion_task_t task, unsigned idx);

  /**
   * @see Legion::Task::futures
   */
  unsigned
  legion_task_get_futures_size(legion_task_t task);

  /**
   * @see Legion::Task::futures
   */
  legion_future_t
  legion_task_get_future(legion_task_t task, unsigned idx);

  /**
   * @see Legion::Task::task_id
   */
  legion_task_id_t
  legion_task_get_task_id(legion_task_t task);

  /**
   * @see Legion::Task::target_proc
   */
  legion_processor_t
  legion_task_get_target_proc(legion_task_t task);

  /**
   * @see Legion::Task::variants::name
   */
  const char *
  legion_task_get_name(legion_task_t task);

  // -----------------------------------------------------------------------
  // Inline Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller does **NOT** take ownership of return value.
   *
   * @see Legion::Inline::requirement
   */
  legion_region_requirement_t
  legion_inline_get_requirement(legion_inline_t inline_operation);

  // -----------------------------------------------------------------------
  // Execution Constraints
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value
   * 
   * @see Legion::ExecutionConstraintSet::ExecutionConstraintSet()
   */
  legion_execution_constraint_set_t
  legion_execution_constraint_set_create(void);

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::ExecutionConstraintSet::~ExecutionConstraintSet()
   */
  void
  legion_execution_constraint_set_destroy(
    legion_execution_constraint_set_t handle);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(Legion::ISAConstraint)
   */
  void
  legion_execution_constraint_set_add_isa_constraint(
    legion_execution_constraint_set_t handle,
    uint64_t prop);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::ProcessorConstraint)
   */
  void
  legion_execution_constraint_set_add_processor_constraint(
    legion_execution_constraint_set_t handle,
    legion_processor_kind_t proc_kind);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::ResourceConstraint)
   */
  void
  legion_execution_constraint_set_add_resource_constraint(
    legion_execution_constraint_set_t handle,
    legion_resource_constraint_t resource,
    legion_equality_kind_t eq,
    size_t value);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::LaunchConstraint)
   */
  void
  legion_execution_constraint_set_add_launch_constraint(
    legion_execution_constraint_set_t handle,
    legion_launch_constraint_t kind,
    size_t value);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::LaunchConstraint)
   */
  void
  legion_execution_constraint_set_add_launch_constraint_multi_dim(
    legion_execution_constraint_set_t handle,
    legion_launch_constraint_t kind,
    const size_t *values,
    int dims);

  /**
   * @see Legion::ExecutionConstraintSet::add_constraint(
   *        Legion::ColocationConstraint)
   */
  void
  legion_execution_constraint_set_add_colocation_constraint(
    legion_execution_constraint_set_t handle,
    const unsigned *indexes,
    size_t num_indexes,
    const legion_field_id_t *fields,
    size_t num_fields);

  // -----------------------------------------------------------------------
  // Layout Constraints
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::LayoutConstraintSet::LayoutConstraintSet()
   */
  legion_layout_constraint_set_t
  legion_layout_constraint_set_create(void);

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::LayoutConstraintSet::~LayoutConstraintSet()
   */
  void
  legion_layout_constraint_set_destroy(legion_layout_constraint_set_t handle);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::register_layout()
   */
  legion_layout_constraint_id_t
  legion_layout_constraint_set_register(
    legion_runtime_t runtime,
    legion_field_space_t fspace,
    legion_layout_constraint_set_t handle,
    const char *layout_name /* = NULL */);

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::Runtime::preregister_layout()
   */
  legion_layout_constraint_id_t
  legion_layout_constraint_set_preregister(
    legion_layout_constraint_set_t handle,
    const char *layout_name /* = NULL */);

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::Runtime::release_layout()
   */
  void
  legion_layout_constraint_set_release(legion_runtime_t runtime,
                                       legion_layout_constraint_id_t handle);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::SpecializedConstraint)
   */
  void
  legion_layout_constraint_set_add_specialized_constraint(
    legion_layout_constraint_set_t handle,
    legion_specialized_constraint_t specialized,
    legion_reduction_op_id_t redop);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::MemoryConstraint)
   */
  void
  legion_layout_constraint_set_add_memory_constraint(
    legion_layout_constraint_set_t handle,
    legion_memory_kind_t kind);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::FieldConstraint)
   */
  void
  legion_layout_constraint_set_add_field_constraint(
    legion_layout_constraint_set_t handle,
    const legion_field_id_t *fields,
    size_t num_fields,
    bool contiguous,
    bool inorder);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::OrderingConstraint)
   */
  void
  legion_layout_constraint_set_add_ordering_constraint(
    legion_layout_constraint_set_t handle,
    const legion_dimension_kind_t *dims,
    size_t num_dims,
    bool contiguous);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::SplittingConstraint)
   */
  void
  legion_layout_constraint_set_add_splitting_constraint(
    legion_layout_constraint_set_t handle,
    legion_dimension_kind_t dim);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::SplittingConstraint)
   */
  void
  legion_layout_constraint_set_add_full_splitting_constraint(
    legion_layout_constraint_set_t handle,
    legion_dimension_kind_t dim,
    size_t value);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::DimensionConstraint)
   */
  void
  legion_layout_constraint_set_add_dimension_constraint(
    legion_layout_constraint_set_t handle,
    legion_dimension_kind_t dim,
    legion_equality_kind_t eq,
    size_t value);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::AlignmentConstraint)
   */
  void
  legion_layout_constraint_set_add_alignment_constraint(
    legion_layout_constraint_set_t handle,
    legion_field_id_t field,
    legion_equality_kind_t eq,
    size_t byte_boundary);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::OffsetConstraint)
   */
  void
  legion_layout_constraint_set_add_offset_constraint(
    legion_layout_constraint_set_t handle,
    legion_field_id_t field,
    size_t offset);

  /**
   * @see Legion::LayoutConstraintSet::add_constraint(
   *        Legion::PointerConstraint)
   */
  void
  legion_layout_constraint_set_add_pointer_constraint(
    legion_layout_constraint_set_t handle,
    legion_memory_t memory,
    uintptr_t ptr);

  // -----------------------------------------------------------------------
  // Task Layout Constraints
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value
   *
   * @see Legion::TaskLayoutConstraintSet::TaskLayoutConstraintSet()
   */
  legion_task_layout_constraint_set_t
  legion_task_layout_constraint_set_create(void);

  /**
   * @param handle Caller must have ownership of parameter 'handle'
   *
   * @see Legion::TaskLayoutConstraintSet::~TaskLayoutConstraintSet()
   */
  void
  legion_task_layout_constraint_set_destroy(
    legion_task_layout_constraint_set_t handle);

  /**
   * @see Legion::TaskLayoutConstraintSet::add_layout_constraint()
   */
  void
  legion_task_layout_constraint_set_add_layout_constraint(
    legion_task_layout_constraint_set_t handle,
    unsigned idx,
    legion_layout_constraint_id_t layout);

  // -----------------------------------------------------------------------
  // Start-up Operations
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Runtime::start()
   */
  int
  legion_runtime_start(int argc,
                       char **argv,
                       bool background /* = false */);

  /**
   * @see Legion::Runtime::wait_for_shutdown()
   */
  void
  legion_runtime_wait_for_shutdown(void);

  /**
   * @see Legion::Runtime::set_top_level_task_id()
   */
  void
  legion_runtime_set_top_level_task_id(legion_task_id_t top_id);

  /**
   * @see Legion::Runtime::get_input_args()
   */
  const legion_input_args_t
  legion_runtime_get_input_args(void);

  /**
   * @see Legion::Runtime::add_registration_callback()
   */
  void
  legion_runtime_add_registration_callback(
    legion_registration_callback_pointer_t callback);

  /**
   * @see Legion::Runtime::generate_library_mapper_ids()
   */
  legion_mapper_id_t
  legion_runtime_generate_library_mapper_ids(
      legion_runtime_t runtime,
      const char *library_name,
      size_t count);

  /**
   * @see Legion::Runtime::replace_default_mapper()
   */
  void
  legion_runtime_replace_default_mapper(
    legion_runtime_t runtime,
    legion_mapper_t mapper,
    legion_processor_t proc);

  /**
   * @see Legion::Runtime::generate_library_projection_ids()
   */
  legion_projection_id_t
  legion_runtime_generate_library_projection_ids(
      legion_runtime_t runtime,
      const char *library_name,
      size_t count);

  /**
   * @see Legion::Runtime::preregister_projection_functor()
   */
  void
  legion_runtime_preregister_projection_functor(
    legion_projection_id_t id,
    unsigned depth,
    legion_projection_functor_logical_region_t region_functor,
    legion_projection_functor_logical_partition_t partition_functor);

  /**
   * @see Legion::Runtime::register_projection_functor()
   */
  void
  legion_runtime_register_projection_functor(
    legion_runtime_t runtime,
    legion_projection_id_t id,
    unsigned depth,
    legion_projection_functor_logical_region_t region_functor,
    legion_projection_functor_logical_partition_t partition_functor);

  /**
   * @see Legion::Runtime::generate_library_task_ids()
   */
  legion_task_id_t
  legion_runtime_generate_library_task_ids(
      legion_runtime_t runtime,
      const char *library_name,
      size_t count);

  /**
   * @see Legion::Runtime::register_task_variant()
   */
  legion_task_id_t
  legion_runtime_register_task_variant_fnptr(
    legion_runtime_t runtime,
    legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    bool global,
    legion_execution_constraint_set_t execution_constraints,
    legion_task_layout_constraint_set_t layout_constraints,
    legion_task_config_options_t options,
    legion_task_pointer_wrapped_t wrapped_task_pointer,
    const void *userdata,
    size_t userlen);

  /**
   * @see Legion::Runtime::preregister_task_variant()
   */
  legion_task_id_t
  legion_runtime_preregister_task_variant_fnptr(
    legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    legion_execution_constraint_set_t execution_constraints,
    legion_task_layout_constraint_set_t layout_constraints,
    legion_task_config_options_t options,
    legion_task_pointer_wrapped_t wrapped_task_pointer,
    const void *userdata,
    size_t userlen);

  /**
   * @see Legion::Runtime::register_task_variant()
   */
  legion_task_id_t
  legion_runtime_register_task_variant_llvmir(
    legion_runtime_t runtime,
    legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    bool global,
    legion_execution_constraint_set_t execution_constraints,
    legion_task_layout_constraint_set_t layout_constraints,
    legion_task_config_options_t options,
    const char *llvmir,
    const char *entry_symbol,
    const void *userdata,
    size_t userlen);

  /**
   * @see Legion::Runtime::preregister_task_variant()
   */
  legion_task_id_t
  legion_runtime_preregister_task_variant_llvmir(
    legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    legion_execution_constraint_set_t execution_constraints,
    legion_task_layout_constraint_set_t layout_constraints,
    legion_task_config_options_t options,
    const char *llvmir,
    const char *entry_symbol,
    const void *userdata,
    size_t userlen);

  /**
   * @see Legion::Runtime::register_task_variant()
   */
  legion_task_id_t
  legion_runtime_register_task_variant_python_source(
    legion_runtime_t runtime,
    legion_task_id_t id /* = AUTO_GENERATE_ID */,
    const char *task_name /* = NULL*/,
    bool global,
    legion_execution_constraint_set_t execution_constraints,
    legion_task_layout_constraint_set_t layout_constraints,
    legion_task_config_options_t options,
    const char *module_name,
    const char *function_name,
    const void *userdata,
    size_t userlen);

  /**
   * @see Legion::LegionTaskWrapper::legion_task_preamble()
   */
  void
  legion_task_preamble(
    const void *data,
    size_t datalen,
    realm_id_t proc_id,
    legion_task_t *taskptr,
    const legion_physical_region_t **regionptr,
    unsigned * num_regions_ptr,
    legion_context_t * ctxptr,
    legion_runtime_t * runtimeptr);

  /**
   * @see Legion::LegionTaskWrapper::legion_task_postamble()
   */
  void
  legion_task_postamble(
    legion_runtime_t runtime,
    legion_context_t ctx,
    const void *retval,
    size_t retsize);

  // -----------------------------------------------------------------------
  // Timing Operations
  // -----------------------------------------------------------------------

  /**
   * @see Realm::Clock::get_current_time_in_micros()
   */
  unsigned long long
  legion_get_current_time_in_micros(void);

  /**
   * @see Realm::Clock::get_current_time_in_nanos()
   */
  unsigned long long
  legion_get_current_time_in_nanos(void);

  /**
   * @see Legion::Runtime::get_current_time()
   */
  legion_future_t
  legion_issue_timing_op_seconds(legion_runtime_t runtime,
                                 legion_context_t ctx);

  /**
   * @see Legion::Runtime::get_current_time_in_microseconds()
   */
  legion_future_t
  legion_issue_timing_op_microseconds(legion_runtime_t runtime,
                                      legion_context_t ctx);

  /**
   * @see Legion::Runtime::get_current_time_in_nanoseconds()
   */
  legion_future_t
  legion_issue_timing_op_nanoseconds(legion_runtime_t runtime,
                                     legion_context_t ctx);

  // -----------------------------------------------------------------------
  // Machine Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::get_machine()
   */
  legion_machine_t
  legion_machine_create();

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Realm::Machine::~Machine()
   */
  void
  legion_machine_destroy(legion_machine_t handle);

  /**
   * @see Realm::Machine::get_all_processors()
   */
  void
  legion_machine_get_all_processors(
    legion_machine_t machine,
    legion_processor_t *processors,
    size_t processors_size);

  /**
   * @see Realm::Machine::get_all_processors()
   */
  size_t
  legion_machine_get_all_processors_size(legion_machine_t machine);

  /**
   * @see Realm::Machine::get_all_memories()
   */
  void
  legion_machine_get_all_memories(
    legion_machine_t machine,
    legion_memory_t *memories,
    size_t memories_size);

  /**
   * @see Realm::Machine::get_all_memories()
   */
  size_t
  legion_machine_get_all_memories_size(legion_machine_t machine);

  // -----------------------------------------------------------------------
  // Processor Operations
  // -----------------------------------------------------------------------

  /**
   * @see Realm::Processor::kind()
   */
  legion_processor_kind_t
  legion_processor_kind(legion_processor_t proc);

  /**
   * @see Realm::Processor::address_space()
   */
  legion_address_space_t
  legion_processor_address_space(legion_processor_t proc);

  // -----------------------------------------------------------------------
  // Memory Operations
  // -----------------------------------------------------------------------

  /**
   * @see Realm::Memory::kind()
   */
  legion_memory_kind_t
  legion_memory_kind(legion_memory_t mem);

  /**
   * @see Realm::Memory::address_space()
   */
  legion_address_space_t
  legion_memory_address_space(legion_memory_t mem);

  // -----------------------------------------------------------------------
  // Processor Query Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::ProcessorQuery::ProcessorQuery()
   */
  legion_processor_query_t
  legion_processor_query_create(legion_machine_t machine);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::ProcessorQuery::ProcessorQuery()
   */
  legion_processor_query_t
  legion_processor_query_create_copy(legion_processor_query_t query);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Realm::Machine::ProcessorQuery::~ProcessorQuery()
   */
  void
  legion_processor_query_destroy(legion_processor_query_t handle);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::only_kind()
   */
  void
  legion_processor_query_only_kind(legion_processor_query_t query,
                                   legion_processor_kind_t kind);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::local_address_space()
   */
  void
  legion_processor_query_local_address_space(legion_processor_query_t query);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::same_address_space_as()
   */
  void
  legion_processor_query_same_address_space_as_processor(legion_processor_query_t query,
                                                         legion_processor_t proc);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::same_address_space_as()
   */
  void
  legion_processor_query_same_address_space_as_memory(legion_processor_query_t query,
                                                      legion_memory_t mem);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::has_affinity_to()
   */
  void
  legion_processor_query_has_affinity_to_memory(legion_processor_query_t query,
                                                legion_memory_t mem,
                                                unsigned min_bandwidth /* = 0 */,
                                                unsigned max_latency /* = 0 */);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::ProcessorQuery::best_affinity_to()
   */
  void
  legion_processor_query_best_affinity_to_memory(legion_processor_query_t query,
                                                 legion_memory_t mem,
                                                 int bandwidth_weight /* = 0 */,
                                                 int latency_weight /* = 0 */);

  /**
   * @see Realm::Machine::ProcessorQuery::count()
   */
  size_t
  legion_processor_query_count(legion_processor_query_t query);

  /**
   * @see Realm::Machine::ProcessorQuery::first()
   */
  legion_processor_t
  legion_processor_query_first(legion_processor_query_t query);

  /**
   * @see Realm::Machine::ProcessorQuery::next()
   */
  legion_processor_t
  legion_processor_query_next(legion_processor_query_t query,
                              legion_processor_t after);

  /**
   * @see Realm::Machine::ProcessorQuery::random()
   */
  legion_processor_t
  legion_processor_query_random(legion_processor_query_t query);

  // -----------------------------------------------------------------------
  // Memory Query Operations
  // -----------------------------------------------------------------------

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::MemoryQuery::MemoryQuery()
   */
  legion_memory_query_t
  legion_memory_query_create(legion_machine_t machine);

  /**
   * @return Caller takes ownership of return value.
   *
   * @see Realm::Machine::MemoryQuery::MemoryQuery()
   */
  legion_memory_query_t
  legion_memory_query_create_copy(legion_memory_query_t query);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Realm::Machine::MemoryQuery::~MemoryQuery()
   */
  void
  legion_memory_query_destroy(legion_memory_query_t handle);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::only_kind()
   */
  void
  legion_memory_query_only_kind(legion_memory_query_t query,
                                legion_memory_kind_t kind);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::local_address_space()
   */
  void
  legion_memory_query_local_address_space(legion_memory_query_t query);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::same_address_space_as()
   */
  void
  legion_memory_query_same_address_space_as_processor(legion_memory_query_t query,
                                                      legion_processor_t proc);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::same_address_space_as()
   */
  void
  legion_memory_query_same_address_space_as_memory(legion_memory_query_t query,
                                                   legion_memory_t mem);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::has_affinity_to()
   */
  void
  legion_memory_query_has_affinity_to_processor(legion_memory_query_t query,
                                                legion_processor_t proc,
                                                unsigned min_bandwidth /* = 0 */,
                                                unsigned max_latency /* = 0 */);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::has_affinity_to()
   */
  void
  legion_memory_query_has_affinity_to_memory(legion_memory_query_t query,
                                             legion_memory_t mem,
                                             unsigned min_bandwidth /* = 0 */,
                                             unsigned max_latency /* = 0 */);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::best_affinity_to()
   */
  void
  legion_memory_query_best_affinity_to_processor(legion_memory_query_t query,
                                                 legion_processor_t proc,
                                                 int bandwidth_weight /* = 0 */,
                                                 int latency_weight /* = 0 */);

  /**
   * Note: Mutates `query`.
   *
   * @see Realm::Machine::MemoryQuery::best_affinity_to()
   */
  void
  legion_memory_query_best_affinity_to_memory(legion_memory_query_t query,
                                              legion_memory_t mem,
                                              int bandwidth_weight /* = 0 */,
                                              int latency_weight /* = 0 */);

  /**
   * @see Realm::Machine::MemoryQuery::count()
   */
  size_t
  legion_memory_query_count(legion_memory_query_t query);

  /**
   * @see Realm::Machine::MemoryQuery::first()
   */
  legion_memory_t
  legion_memory_query_first(legion_memory_query_t query);

  /**
   * @see Realm::Machine::MemoryQuery::next()
   */
  legion_memory_t
  legion_memory_query_next(legion_memory_query_t query,
                           legion_memory_t after);

  /**
   * @see Realm::Machine::MemoryQuery::random()
   */
  legion_memory_t
  legion_memory_query_random(legion_memory_query_t query);

  // -----------------------------------------------------------------------
  // Physical Instance Operations
  // -----------------------------------------------------------------------

  /*
   * @param instance Caller must have ownership of parameter `instance`.
   *
   * @see Legion::Mapping::PhysicalInstance
   */
  void
  legion_physical_instance_destroy(legion_physical_instance_t instance);

  // -----------------------------------------------------------------------
  // Slice Task Output
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Mapping::Mapper::SliceTaskOutput:slices
   */
  void
  legion_slice_task_output_slices_add(
      legion_slice_task_output_t output,
      legion_task_slice_t slice);

  /**
   * @see Legion::Mapping::Mapper::SliceTaskOutput:verify_correctness
   */
  void
  legion_slice_task_output_verify_correctness_set(
      legion_slice_task_output_t output,
      bool verify_correctness);

  // -----------------------------------------------------------------------
  // Map Task Input/Output
  // -----------------------------------------------------------------------

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:chosen_instances
   */
  void
  legion_map_task_output_chosen_instances_clear_all(
      legion_map_task_output_t output);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:chosen_instances
   */
  void
  legion_map_task_output_chosen_instances_clear_each(
      legion_map_task_output_t output,
      size_t idx);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:chosen_instances
   */
  void
  legion_map_task_output_chosen_instances_add(
      legion_map_task_output_t output,
      legion_physical_instance_t *instances,
      size_t instances_size);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:chosen_instances
   */
  void
  legion_map_task_output_chosen_instances_set(
      legion_map_task_output_t output,
      size_t idx,
      legion_physical_instance_t *instances,
      size_t instances_size);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:target_procs
   */
  void
  legion_map_task_output_target_procs_clear(
      legion_map_task_output_t output);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:target_procs
   */
  void
  legion_map_task_output_target_procs_add(
      legion_map_task_output_t output,
      legion_processor_t proc);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:target_procs
   */
  legion_processor_t
  legion_map_task_output_target_procs_get(
      legion_map_task_output_t output,
      size_t idx);

  /**
   * @see Legion::Mapping::Mapper::MapTaskOutput:task_priority
   */
  void
  legion_map_task_output_task_priority_set(
      legion_map_task_output_t output,
      legion_task_priority_t priority);

  // -----------------------------------------------------------------------
  // MapperRuntime Operations
  // -----------------------------------------------------------------------

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::create_physical_instance()
   */
  bool
  legion_mapper_runtime_create_physical_instance_layout_constraint(
      legion_mapper_runtime_t runtime,
      legion_mapper_context_t ctx,
      legion_memory_t target_memory,
      legion_layout_constraint_set_t constraints,
      const legion_logical_region_t *regions,
      size_t regions_size,
      legion_physical_instance_t *result,
      bool acquire,
      legion_garbage_collection_priority_t priority);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::create_physical_instance()
   */
  bool
  legion_mapper_runtime_create_physical_instance_layout_constraint_id(
      legion_mapper_runtime_t runtime,
      legion_mapper_context_t ctx,
      legion_memory_t target_memory,
      legion_layout_constraint_id_t layout_id,
      const legion_logical_region_t *regions,
      size_t regions_size,
      legion_physical_instance_t *result,
      bool acquire,
      legion_garbage_collection_priority_t priority);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::find_or_create_physical_instance()
   */
  bool
  legion_mapper_runtime_find_or_create_physical_instance_layout_constraint(
      legion_mapper_runtime_t runtime,
      legion_mapper_context_t ctx,
      legion_memory_t target_memory,
      legion_layout_constraint_set_t constraints,
      const legion_logical_region_t *regions,
      size_t regions_size,
      legion_physical_instance_t *result,
      bool *created,
      bool acquire,
      legion_garbage_collection_priority_t priority,
      bool tight_region_bounds);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::find_or_create_physical_instance()
   */
  bool
  legion_mapper_runtime_find_or_create_physical_instance_layout_constraint_id(
      legion_mapper_runtime_t runtime,
      legion_mapper_context_t ctx,
      legion_memory_t target_memory,
      legion_layout_constraint_id_t layout_id,
      const legion_logical_region_t *regions,
      size_t regions_size,
      legion_physical_instance_t *result,
      bool *created,
      bool acquire,
      legion_garbage_collection_priority_t priority,
      bool tight_region_bounds);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::find_physical_instance()
   */
  bool
  legion_mapper_runtime_find_physical_instance_layout_constraint(
      legion_mapper_runtime_t runtime,
      legion_mapper_context_t ctx,
      legion_memory_t target_memory,
      legion_layout_constraint_set_t constraints,
      const legion_logical_region_t *regions,
      size_t regions_size,
      legion_physical_instance_t *result,
      bool acquire,
      bool tight_region_bounds);

  /**
   * @param result Caller takes ownership of handle pointed by `result`.
   *
   * @see Legion::Mapping::MapperRuntime::find_physical_instance()
   */
  bool
  legion_mapper_runtime_find_physical_instance_layout_constraint_id(
      legion_mapper_runtime_t runtime,
      legion_mapper_context_t ctx,
      legion_memory_t target_memory,
      legion_layout_constraint_id_t layout_id,
      const legion_logical_region_t *regions,
      size_t regions_size,
      legion_physical_instance_t *result,
      bool acquire,
      bool tight_region_bounds);


  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Mapping::MapperRuntime::acquire_instance()
   */
  bool
  legion_mapper_runtime_acquire_instance(
      legion_mapper_runtime_t runtime,
      legion_mapper_context_t ctx,
      legion_physical_instance_t instance);

  /**
   * @param handle Caller must have ownership of parameter `handle`.
   *
   * @see Legion::Mapping::MapperRuntime::acquire_instances()
   */
  bool
  legion_mapper_runtime_acquire_instances(
      legion_mapper_runtime_t runtime,
      legion_mapper_context_t ctx,
      legion_physical_instance_t *instances,
      size_t instances_size);

#ifdef __cplusplus
}
#endif

#endif // __LEGION_C_H__
