! Copyright 2022 Stanford University, NVIDIA Corporation,
!                Los Alamos National Laboratory
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

module legion_fortran_types
  use, intrinsic :: iso_c_binding
  implicit none

#include "legion_defines.h"

#ifndef LEGION_MAX_DIM
#define LEGION_MAX_DIM     3
#endif

#if LEGION_MAX_DIM >= 4
#error "Illegal value of LEGION_MAX_DIM"
#endif

#define MAX_POINT_DIM_F LEGION_MAX_DIM
#define MAX_RECT_DIM_F LEGION_MAX_DIM

  ! legion_privilege_mode_t
  integer(c_int), parameter :: NO_ACCESS = int(Z'00000000', c_int)
  integer(c_int), parameter :: READ_PRIV = int(Z'00000001', c_int)
  integer(c_int), parameter :: READ_ONLY = int(Z'00000001', c_int)
  integer(c_int), parameter :: WRITE_PRIV = int(Z'00000002', c_int)
  integer(c_int), parameter :: REDUCE_PRIV = int(Z'00000004', c_int)
  integer(c_int), parameter :: REDUCE = int(Z'00000004', c_int)
  integer(c_int), parameter :: READ_WRITE = int(Z'00000007', c_int)
  integer(c_int), parameter :: DISCARD_MASK = int(Z'10000000', c_int)
  integer(c_int), parameter :: WRITE_ONLY = int(Z'10000002', c_int)
  integer(c_int), parameter :: WRITE_DISCARD = int(Z'10000007', c_int)

  ! legion_coherence_property_t
  integer(c_int), parameter :: EXCLUSIVE = 0
  integer(c_int), parameter :: ATOMIC = 1
  integer(c_int), parameter :: SIMULTANEOUS = 2
  integer(c_int), parameter :: RELAXED = 3

  !legion_file_mode_t
  integer(c_int), parameter :: LEGION_FILE_READ_ONLY = 0
  integer(c_int), parameter :: LEGION_FILE_READ_WRITE = 1
  integer(c_int), parameter :: LEGION_FILE_CREATE = 2

  !legion_processor_kind_t
  integer(c_int), parameter :: NO_KIND = 0
  integer(c_int), parameter :: TOC_PROC = 1
  integer(c_int), parameter :: LOC_PROC = 2
  integer(c_int), parameter :: UTIL_PROC = 3
  integer(c_int), parameter :: IO_PROC = 4
  integer(c_int), parameter :: PROC_GROUP = 5
  integer(c_int), parameter :: PROC_SET = 6
  integer(c_int), parameter :: OMP_PROC = 7
  integer(c_int), parameter :: PY_PROC = 8

  ! legion_partition_kind_t
  integer(c_int), parameter :: DISJOINT_KIND = 0
  integer(c_int), parameter :: ALIASED_KIND = 1
  integer(c_int), parameter :: COMPUTE_KIND = 2
  integer(c_int), parameter :: DISJOINT_COMPLETE_KIND = 3
  integer(c_int), parameter :: ALIASED_COMPLETE_KIND = 4
  integer(c_int), parameter :: COMPUTE_COMPLETE_KIND = 5
  integer(c_int), parameter :: DISJOINT_INCOMPLETE_KIND = 6
  integer(c_int), parameter :: ALIASED_INCOMPLETE_KIND = 7
  integer(c_int), parameter :: COMPUTE_INCOMPLETE_KIND = 8

  ! legion_external_resource_t
  integer(c_int), parameter :: EXTERNAL_POSIX_FILE = 0
  integer(c_int), parameter :: EXTERNAL_HDF5_FILE = 1
  integer(c_int), parameter :: EXTERNAL_INSTANCE = 2

  integer(c_int), parameter :: I4_MAX = 1
  integer(c_int), parameter :: AUTO_GENERATE_ID = -1 !huge(I4_MAX)

    ! C NEW_OPAQUE_TYPE_F
#define NEW_OPAQUE_TYPE_F(T) type, bind(C) :: T; type(c_ptr) :: impl; end type T
  NEW_OPAQUE_TYPE_F(legion_runtime_f_t)
  NEW_OPAQUE_TYPE_F(legion_context_f_t)
  NEW_OPAQUE_TYPE_F(legion_domain_point_iterator_f_t)
  NEW_OPAQUE_TYPE_F(legion_coloring_f_t)
  NEW_OPAQUE_TYPE_F(legion_domain_coloring_f_t)
  NEW_OPAQUE_TYPE_F(legion_point_coloring_f_t)
  NEW_OPAQUE_TYPE_F(legion_domain_point_coloring_f_t)
  NEW_OPAQUE_TYPE_F(legion_multi_domain_point_coloring_f_t)
  NEW_OPAQUE_TYPE_F(legion_index_space_allocator_f_t)
  NEW_OPAQUE_TYPE_F(legion_field_allocator_f_t)
  NEW_OPAQUE_TYPE_F(legion_argument_map_f_t)
  NEW_OPAQUE_TYPE_F(legion_predicate_f_t)
  NEW_OPAQUE_TYPE_F(legion_future_f_t)
  NEW_OPAQUE_TYPE_F(legion_future_map_f_t)
  NEW_OPAQUE_TYPE_F(legion_task_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_index_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_inline_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_copy_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_index_copy_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_acquire_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_release_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_attach_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_must_epoch_launcher_f_t)
  NEW_OPAQUE_TYPE_F(legion_physical_region_f_t)
  NEW_OPAQUE_TYPE_F(legion_accessor_array_1d_f_t)
  NEW_OPAQUE_TYPE_F(legion_accessor_array_2d_f_t)
  NEW_OPAQUE_TYPE_F(legion_accessor_array_3d_f_t)
  NEW_OPAQUE_TYPE_F(legion_index_iterator_f_t)
  NEW_OPAQUE_TYPE_F(legion_task_f_t)
  NEW_OPAQUE_TYPE_F(legion_inline_f_t)
  NEW_OPAQUE_TYPE_F(legion_mappable_f_t)
  NEW_OPAQUE_TYPE_F(legion_region_requirement_f_t)
  NEW_OPAQUE_TYPE_F(legion_machine_f_t)
  NEW_OPAQUE_TYPE_F(legion_mapper_f_t)
  NEW_OPAQUE_TYPE_F(legion_default_mapper_f_t)
  NEW_OPAQUE_TYPE_F(legion_processor_query_f_t)
  NEW_OPAQUE_TYPE_F(legion_memory_query_f_t)
  NEW_OPAQUE_TYPE_F(legion_machine_query_interface_f_t)
  NEW_OPAQUE_TYPE_F(legion_execution_constraint_set_f_t)
  NEW_OPAQUE_TYPE_F(legion_layout_constraint_set_f_t)
  NEW_OPAQUE_TYPE_F(legion_task_layout_constraint_set_f_t)
  NEW_OPAQUE_TYPE_F(legion_slice_task_output_f_t)
  NEW_OPAQUE_TYPE_F(legion_map_task_input_f_t)
  NEW_OPAQUE_TYPE_F(legion_map_task_output_f_t)
  NEW_OPAQUE_TYPE_F(legion_physical_instance_f_t)
  NEW_OPAQUE_TYPE_F(legion_mapper_runtime_f_t)
  NEW_OPAQUE_TYPE_F(legion_mapper_context_f_t)
  NEW_OPAQUE_TYPE_F(legion_field_map_f_t)
#undef NEW_OPAQUE_TYPE_F

    ! point 1d, 2d, 3d
#define NEW_POINT_TYPE_F(T, DIM) type, bind(C) :: T; integer(c_long_long), dimension(0:DIM-1) :: x; end type T
  NEW_POINT_TYPE_F(legion_point_1d_f_t, 1)
#if LEGION_MAX_DIM >= 2
  NEW_POINT_TYPE_F(legion_point_2d_f_t, 2)
#endif
#if LEGION_MAX_DIM >= 3
  NEW_POINT_TYPE_F(legion_point_3d_f_t, 3)
#endif
#undef NEW_POINT_TYPE_F

! #define LEGION_FOREACH_FN(__func__) __func__(1); __func__(2); __func__(3)
!
! #ifdef __GFORTRAN__
! #define NEW_POINT_TYPE_F(DIM) type, bind(C) :: legion_point_/**/DIM/**/d_f_t; integer(c_long_long), dimension(0:DIM-1) :: x; end type legion_point_/**/DIM/**/d_f_t
!    LEGION_FOREACH_FN(NEW_POINT_TYPE_F)
! #undef NEW_POINT_TYPE_F
! #else
! #define NEW_POINT_TYPE_F(DIM) type, bind(C) :: legion_point_/**/DIM/**/d_f_t; integer(c_long_long), dimension(0:DIM-1) :: x; end type legion_point_/**/DIM/**/d_f_t
!    NEW_POINT_TYPE_F(1)
!    NEW_POINT_TYPE_F(2)
!    NEW_POINT_TYPE_F(3)
! #undef NEW_POINT_TYPE_F
! #endif

    ! rect 1d, 2d, 3d
#define NEW_RECT_TYPE_F(T, PT) type, bind(C) :: T; type(PT) :: lo, hi; end type T
  NEW_RECT_TYPE_F(legion_rect_1d_f_t, legion_point_1d_f_t)
#if LEGION_MAX_DIM >= 2
  NEW_RECT_TYPE_F(legion_rect_2d_f_t, legion_point_2d_f_t)
#endif
#if LEGION_MAX_DIM >= 3
  NEW_RECT_TYPE_F(legion_rect_3d_f_t, legion_point_3d_f_t)
#endif

    ! transform 1x1,2x2,3x3
#define NEW_TRANSFORM_TYPE_F(T, D1, D2) type, bind(C) :: T; integer(c_long_long) :: trans(0:D1-1, 0:D2-1); end type T
  NEW_TRANSFORM_TYPE_F(legion_transform_1x1_f_t, 1, 1)
#if LEGION_MAX_DIM >= 2
  NEW_TRANSFORM_TYPE_F(legion_transform_1x2_f_t, 1, 2)
  NEW_TRANSFORM_TYPE_F(legion_transform_2x1_f_t, 2, 1)
  NEW_TRANSFORM_TYPE_F(legion_transform_2x2_f_t, 2, 2)
#endif
#if LEGION_MAX_DIM >= 3
  NEW_TRANSFORM_TYPE_F(legion_transform_1x3_f_t, 1, 3)
  NEW_TRANSFORM_TYPE_F(legion_transform_2x3_f_t, 2, 3)
  NEW_TRANSFORM_TYPE_F(legion_transform_3x1_f_t, 3, 1)
  NEW_TRANSFORM_TYPE_F(legion_transform_3x2_f_t, 3, 2)
  NEW_TRANSFORM_TYPE_F(legion_transform_3x3_f_t, 3, 3)
#endif

  ! Legion::Domain
  type, bind(C) :: legion_domain_f_t
    integer(c_long_long)                                  :: is_id
    integer(c_int)                                        :: dim
#if LEGION_MAX_DIM == 1
#define MAX_DOMAIN_DIM_F 2
#elif LEGION_MAX_DIM == 2
#define MAX_DOMAIN_DIM_F 4
#elif LEGION_MAX_DIM == 3
#define MAX_DOMAIN_DIM_F 6
#elif LEGION_MAX_DIM == 4
#define MAX_DOMAIN_DIM_F 8
#elif LEGION_MAX_DIM == 5
#define MAX_DOMAIN_DIM_F 10
#elif LEGION_MAX_DIM == 6
#define MAX_DOMAIN_DIM_F 12
#elif LEGION_MAX_DIM == 7
#define MAX_DOMAIN_DIM_F 14
#elif LEGION_MAX_DIM == 8
#define MAX_DOMAIN_DIM_F 16
#elif LEGION_MAX_DIM == 9
#define MAX_DOMAIN_DIM_F 18
#else
#error "Illegal value of LEGION_MAX_DIM"
#endif
    integer(c_long_long), dimension(0:MAX_DOMAIN_DIM_F-1) :: rect_data
#undef MAX_DOMAIN_DIM_F
  end type legion_domain_f_t

  ! Legion::DomainPoint
  type, bind(C) :: legion_domain_point_f_t
    integer(c_int)                                       :: dim
    integer(c_long_long), dimension(0:MAX_POINT_DIM_F-1) :: point_data
  end type legion_domain_point_f_t

  ! Legion::Transform
  type, bind(C) :: legion_domain_transform_f_t
    integer(c_int)                                        :: m
    integer(c_int)                                        :: n
#if LEGION_MAX_DIM == 1
#define MAX_MATRIX_DIM_F 1
#elif LEGION_MAX_DIM == 2
#define MAX_MATRIX_DIM_F 4
#elif LEGION_MAX_DIM == 3
#define MAX_MATRIX_DIM_F 9
#elif LEGION_MAX_DIM == 4
#define MAX_MATRIX_DIM_F 16
#elif LEGION_MAX_DIM == 5
#define MAX_MATRIX_DIM_F 25
#elif LEGION_MAX_DIM == 6
#define MAX_MATRIX_DIM_F 36
#elif LEGION_MAX_DIM == 7
#define MAX_MATRIX_DIM_F 49
#elif LEGION_MAX_DIM == 8
#define MAX_MATRIX_DIM_F 64
#elif LEGION_MAX_DIM == 9
#define MAX_MATRIX_DIM_F 81
#else
#error "Illegal value of LEGION_MAX_DIM"
#endif
    integer(c_long_long), dimension(0:MAX_MATRIX_DIM_F-1) :: matrix
#undef MAX_MATRIX_DIM_F
  end type legion_domain_transform_f_t

  ! Legion::IndexSpace
  type, bind(C) :: legion_index_space_f_t
    integer(c_int) :: id
    integer(c_int) :: tid
    integer(c_int) :: type_tag
  end type legion_index_space_f_t

  ! Legion::IndexPartition
  type, bind(C) :: legion_index_partition_f_t
    integer(c_int) :: id
    integer(c_int) :: tid
    integer(c_int) :: type_tag
  end type legion_index_partition_f_t

  ! Legion::FieldSpace
  type, bind(C) :: legion_field_space_f_t
    integer(c_int) :: id
  end type legion_field_space_f_t

  ! Legion::LogicalRegion
  type, bind(C) :: legion_logical_region_f_t
    integer(c_int)               :: tree_id
    type(legion_index_space_f_t) :: index_space
    type(legion_field_space_f_t) :: field_space
  end type legion_logical_region_f_t

  ! Legion::LogicalPartition
  type, bind(C) :: legion_logical_partition_f_t
    integer(c_int)                   :: tree_id
    type(legion_index_partition_f_t) :: index_partition
    type(legion_field_space_f_t)     :: field_space
  end type legion_logical_partition_f_t

  ! Legion::TaskConfigOptions
  type, bind(C) :: legion_task_config_options_f_t
    logical(c_bool) :: leaf
    logical(c_bool) :: inner
    logical(c_bool) :: idempotent
    logical(c_bool) :: replicable
  end type legion_task_config_options_f_t

  ! Legion::UntypedBuffer
  type, bind(C) :: legion_task_argument_f_t
    type(c_ptr)         :: args
    integer(c_size_t)   :: arglen
  end type legion_task_argument_f_t

  ! offest
  type, bind(C) :: legion_byte_offset_f_t
    integer(c_int) :: offset
  end type legion_byte_offset_f_t

    ! C typedef enum
  !  enum, bind(C) :: legion_processor_kind_t
   !     enumrator :: NO_KIND = 0
    !    TOC_PROC, LOC_PROC, UTIL_PROC, IO_PROC, PROC_GROUP, PROC_SET, OMP_PROC
    !end enum
end module
