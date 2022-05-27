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

module legion_fortran_c_interface
  use, intrinsic :: iso_c_binding
  use legion_fortran_types
  implicit none

#include "legion_defines.h"

#ifndef LEGION_MAX_DIM
#define LEGION_MAX_DIM     3
#endif

#if LEGION_MAX_DIM >= 4
#error "Illegal value of LEGION_MAX_DIM"
#endif

  interface
    ! -----------------------------------------------------------------------
    ! Task Launcher
    ! -----------------------------------------------------------------------
    ! @see Legion::TaskLauncher::TaskLauncher()
    function legion_task_launcher_create_f(tid, arg, pred, id, tag) &
        bind(C, name="legion_task_launcher_create")
      use iso_c_binding
      import legion_task_launcher_f_t
      import legion_task_argument_f_t
      import legion_predicate_f_t
      implicit none

      type(legion_task_launcher_f_t)                      :: legion_task_launcher_create_f
      integer(c_int), value, intent(in)                   :: tid
      type(legion_task_argument_f_t), value, intent(in)   :: arg
      type(legion_predicate_f_t), value, intent(in)       :: pred
      integer(c_int), value, intent(in)                   :: id
      integer(c_long), value, intent(in)                  :: tag
    end function legion_task_launcher_create_f

    ! @see Legion::TaskLauncher::~TaskLauncher()
    subroutine legion_task_launcher_destroy_f(handle) &
        bind(C, name="legion_task_launcher_destroy")
      use iso_c_binding
      import legion_task_launcher_f_t
      implicit none

      type(legion_task_launcher_f_t), value, intent(in) :: handle
    end subroutine legion_task_launcher_destroy_f

    ! @see Legion::Runtime::execute_task()
    function legion_task_launcher_execute_f(runtime, ctx, launcher) &
        bind(C, name="legion_task_launcher_execute")
      use iso_c_binding
      import legion_future_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_task_launcher_f_t
      implicit none

      type(legion_future_f_t)                            :: legion_task_launcher_execute_f
      type(legion_runtime_f_t), value, intent(in)        :: runtime
      type(legion_context_f_t), value, intent(in)        :: ctx
      type(legion_task_launcher_f_t), value, intent(in)  :: launcher
    end function legion_task_launcher_execute_f

    ! @see Legion::TaskLauncher::add_region_requirement()
    function legion_task_launcher_add_region_requirement_logical_region_f( &
        launcher, handle, priv, prop, parent, tag, verified) &
        bind (C, name="legion_task_launcher_add_region_requirement_logical_region")
      use iso_c_binding
      import legion_task_launcher_f_t
      import legion_logical_region_f_t
      implicit none

      integer(c_int)                                     :: legion_task_launcher_add_region_requirement_logical_region_f
      type(legion_task_launcher_f_t), value, intent(in)  :: launcher
      type(legion_logical_region_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                  :: priv
      integer(c_int), value, intent(in)                  :: prop
      type(legion_logical_region_f_t), value, intent(in) :: parent
      integer(c_long), value, intent(in)                 :: tag
      logical(c_bool), value, intent(in)                 :: verified
    end function legion_task_launcher_add_region_requirement_logical_region_f

    ! @see Legion::TaskLaunchxer::add_field()
    subroutine legion_task_launcher_add_field_f(launcher, idx, fid, inst) &
        bind(C, name="legion_task_launcher_add_field")
      use iso_c_binding
      import legion_task_launcher_f_t
      implicit none

      type(legion_task_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                 :: idx
      integer(c_int), value, intent(in)                 :: fid
      logical(c_bool), value, intent(in)                :: inst
    end subroutine legion_task_launcher_add_field_f

    ! @see Legion::TaskLauncher::add_future()
    subroutine legion_task_launcher_add_future_f(launcher, future) &
        bind(C, name="legion_task_launcher_add_future")
      use iso_c_binding
      import legion_task_launcher_f_t
      import legion_future_f_t
      implicit none

      type(legion_task_launcher_f_t), value, intent(in) :: launcher
      type(legion_future_f_t), value, intent(in)        :: future
    end subroutine legion_task_launcher_add_future_f

    ! -----------------------------------------------------------------------
    ! Index Launcher
    ! -----------------------------------------------------------------------
    ! @see Legion::IndexTaskLauncher::IndexTaskLauncher()
    function legion_index_launcher_create_f(tid, domain, global_arg, &
        map, pred, must, id, tag) &
        bind(C, name="legion_index_launcher_create")
      use iso_c_binding
      import legion_index_launcher_f_t
      import legion_domain_f_t
      import legion_task_argument_f_t
      import legion_argument_map_f_t
      import legion_predicate_f_t
      implicit none

      type(legion_index_launcher_f_t)                     :: legion_index_launcher_create_f
      integer(c_int), value, intent(in)                   :: tid
      type(legion_domain_f_t), value, intent(in)          :: domain
      type(legion_task_argument_f_t), value, intent(in)   :: global_arg
      type(legion_argument_map_f_t), value, intent(in)    :: map
      type(legion_predicate_f_t), value, intent(in)       :: pred
      logical(c_bool), value, intent(in)                  :: must
      integer(c_int), value, intent(in)                   :: id
      integer(c_long), value, intent(in)                  :: tag
    end function legion_index_launcher_create_f

    ! @see Legion::IndexTaskLauncher::~IndexTaskLauncher()
    subroutine legion_index_launcher_destroy_f(handle) &
        bind(C, name="legion_index_launcher_destroy")
      use iso_c_binding
      import legion_index_launcher_f_t
      implicit none

      type(legion_index_launcher_f_t), value, intent(in) :: handle
    end subroutine legion_index_launcher_destroy_f

    ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &)
    function legion_index_launcher_execute_f(runtime, ctx, launcher) &
        bind(C, name="legion_index_launcher_execute")
      use iso_c_binding
      import legion_future_map_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_launcher_f_t
      implicit none

      type(legion_future_map_f_t)                         :: legion_index_launcher_execute_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_index_launcher_f_t), value, intent(in)  :: launcher
    end function legion_index_launcher_execute_f

    ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, ReductionOpID)
    function legion_index_launcher_execute_reduction_f(runtime, ctx, &
        launcher, redop) &
        bind(C, name="legion_index_launcher_execute_reduction")
      use iso_c_binding
      import legion_future_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_launcher_f_t
      implicit none

      type(legion_future_f_t)                            :: legion_index_launcher_execute_reduction_f
      type(legion_runtime_f_t), value, intent(in)        :: runtime
      type(legion_context_f_t), value, intent(in)        :: ctx
      type(legion_index_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                  :: redop
    end function legion_index_launcher_execute_reduction_f

    ! @see Legion::IndexTaskLauncher::add_region_requirement()
    function legion_index_launcher_add_region_requirement_lp_f( &
        launcher, handle, proj, priv, &
        prop, parent, tag, verified) &
        bind (C, name="legion_index_launcher_add_region_requirement_logical_partition")
      use iso_c_binding
      import legion_index_launcher_f_t
      import legion_logical_partition_f_t
      import legion_logical_region_f_t
      implicit none

      integer(c_int)                                        :: legion_index_launcher_add_region_requirement_lp_f
      type(legion_index_launcher_f_t), value, intent(in)    :: launcher
      type(legion_logical_partition_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                     :: proj
      integer(c_int), value, intent(in)                     :: priv
      integer(c_int), value, intent(in)                     :: prop
      type(legion_logical_region_f_t), value, intent(in)    :: parent
      integer(c_long), value, intent(in)                    :: tag
      logical(c_bool), value, intent(in)                    :: verified
    end function legion_index_launcher_add_region_requirement_lp_f

    ! @see Legion::TaskLaunchxer::add_field()
    subroutine legion_index_launcher_add_field_f(launcher, idx, fid, inst) &
        bind(C, name="legion_index_launcher_add_field")
      use iso_c_binding
      import legion_index_launcher_f_t
      implicit none

      type(legion_index_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                  :: idx
      integer(c_int), value, intent(in)                  :: fid
      logical(c_bool), value, intent(in)                 :: inst
    end subroutine legion_index_launcher_add_field_f

    ! -----------------------------------------------------------------------
    ! Inline Mapping Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::InlineLauncher::InlineLauncher()
    function legion_inline_launcher_create_logical_region_f( &
        handle, priv, prop, parent, &
        region_tag, verified, id, launcher_tag) &
        bind(C, name="legion_inline_launcher_create_logical_region")
      use iso_c_binding
      import legion_inline_launcher_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_inline_launcher_f_t)                   :: legion_inline_launcher_create_logical_region_f
      type(legion_logical_region_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                  :: priv
      integer(c_int), value, intent(in)                  :: prop
      type(legion_logical_region_f_t), value, intent(in) :: parent
      integer(c_long), value, intent(in)                 :: region_tag
      logical(c_bool), value, intent(in)                 :: verified
      integer(c_long), value, intent(in)                 :: id
      integer(c_long), value, intent(in)                 :: launcher_tag
    end function legion_inline_launcher_create_logical_region_f

    ! @see Legion::InlineLauncher::~InlineLauncher()
    subroutine legion_inline_launcher_destroy_f(handle) &
        bind(C, name="legion_inline_launcher_destroy")
      use iso_c_binding
      import legion_inline_launcher_f_t
      implicit none

      type(legion_inline_launcher_f_t), value, intent(in) :: handle
    end subroutine legion_inline_launcher_destroy_f

    ! @see Legion::Runtime::map_region()
    function legion_inline_launcher_execute_f(runtime, ctx, launcher) &
        bind(C, name="legion_inline_launcher_execute")
      use iso_c_binding
      import legion_physical_region_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_inline_launcher_f_t
      implicit none

      type(legion_physical_region_f_t)                    :: legion_inline_launcher_execute_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_inline_launcher_f_t), value, intent(in) :: launcher
    end function legion_inline_launcher_execute_f

    ! @see Legion::InlineLauncher::add_field()
    subroutine legion_inline_launcher_add_field_f(launcher, fid, inst) &
        bind(C, name="legion_inline_launcher_add_field")
      use iso_c_binding
      import legion_inline_launcher_f_t
      implicit none

      type(legion_inline_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                   :: fid
      logical(c_bool), value, intent(in)                  :: inst
    end subroutine legion_inline_launcher_add_field_f

    ! @see Legion::Runtime::remap_region()
    subroutine legion_runtime_remap_region_f(runtime, ctx, region) &
        bind(C, name="legion_runtime_remap_region")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_physical_region_f_t), value, intent(in) :: region
    end subroutine legion_runtime_remap_region_f

    ! @see Legion::Runtime::unmap_region()
    subroutine legion_runtime_unmap_region_f(runtime, ctx, region) &
        bind(C, name="legion_runtime_unmap_region")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_physical_region_f_t), value, intent(in) :: region
    end subroutine legion_runtime_unmap_region_f

    ! @see Legion::Runtime::unmap_all_regions()
    subroutine legion_runtime_unmap_all_regions_f(runtime, ctx) &
        bind(C, name="legion_runtime_unmap_all_regions")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in) :: runtime
      type(legion_context_f_t), value, intent(in) :: ctx
    end subroutine legion_runtime_unmap_all_regions_f

    ! -----------------------------------------------------------------------
    ! Predicate Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Predicate::TRUE_PRED
    function legion_predicate_true_f() &
        bind(C, name="legion_predicate_true")
      use iso_c_binding
      import legion_predicate_f_t
      implicit none

      type(legion_predicate_f_t) :: legion_predicate_true_f
    end function legion_predicate_true_f

    ! @see Legion::Predicate::FALSE_PRED
    function legion_predicate_false_f() &
        bind(C, name="legion_predicate_false")
      use iso_c_binding
      import legion_predicate_f_t
      implicit none

      type(legion_predicate_f_t) :: legion_predicate_false_f
    end function legion_predicate_false_f

    ! -----------------------------------------------------------------------
    ! Argument Map
    ! -----------------------------------------------------------------------
    ! @see Legion::ArgumentMap::ArgumentMap()
    function legion_argument_map_create_f() &
        bind(C, name="legion_argument_map_create")
      use iso_c_binding
      import legion_argument_map_f_t
      implicit none

      type(legion_argument_map_f_t) :: legion_argument_map_create_f
    end function legion_argument_map_create_f

    ! @see Legion::ArgumentMap::set_point()
    subroutine legion_argument_map_set_point_f(map, dp, arg, replace) &
        bind(C, name="legion_argument_map_set_point")
      use iso_c_binding
      import legion_argument_map_f_t
      import legion_domain_point_f_t
      import legion_task_argument_f_t
      implicit none

      type(legion_argument_map_f_t), value, intent(in)  :: map
      type(legion_domain_point_f_t), value, intent(in)  :: dp
      type(legion_task_argument_f_t), value, intent(in) :: arg
      logical(c_bool), value, intent(in)                :: replace
    end subroutine legion_argument_map_set_point_f

    subroutine legion_argument_map_destroy_f(handle) &
        bind(C, name="legion_argument_map_destroy")
      use iso_c_binding
      import legion_argument_map_f_t
      implicit none

      type(legion_argument_map_f_t), value, intent(in) :: handle
    end subroutine legion_argument_map_destroy_f

    ! -----------------------------------------------------------------------
    ! Task Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Task::args
    function legion_task_get_args_f(task) &
        bind(C, name="legion_task_get_args")
      use iso_c_binding
      import legion_task_f_t
      implicit none

      type(c_ptr)                              :: legion_task_get_args_f
      type(legion_task_f_t), value, intent(in) :: task
    end function legion_task_get_args_f

    ! @see Legion::Task::arglen
    function legion_task_get_arglen_f(task) &
        bind(C, name="legion_task_get_arglen")
      use iso_c_binding
      import legion_task_f_t
      implicit none

      integer(c_size_t)                        :: legion_task_get_arglen_f
      type(legion_task_f_t), value, intent(in) :: task
    end function legion_task_get_arglen_f

    ! @see Legion::Task::local_args
    function legion_task_get_local_args_f(task) &
        bind(C, name="legion_task_get_local_args")
      use iso_c_binding
      import legion_task_f_t
      implicit none

      type(c_ptr)                              :: legion_task_get_local_args_f
      type(legion_task_f_t), value, intent(in) :: task
    end function legion_task_get_local_args_f

    ! @see Legion::Task::local_arglen
    function legion_task_get_local_arglen_f(task) &
        bind(C, name="legion_task_get_local_arglen")
      use iso_c_binding
      import legion_task_f_t
      implicit none

      integer(c_size_t)                        :: legion_task_get_local_arglen_f
      type(legion_task_f_t), value, intent(in) :: task
    end function legion_task_get_local_arglen_f

    ! @see Legion::Task::index_domain
    function legion_task_get_index_domain_f(task) &
        bind(C, name="legion_task_get_index_domain")
      use iso_c_binding
      import legion_domain_f_t
      import legion_task_f_t
      implicit none

      type(legion_domain_f_t)                  :: legion_task_get_index_domain_f
      type(legion_task_f_t), value, intent(in) :: task
    end function legion_task_get_index_domain_f

    ! @see Legion::Task::regions
    function legion_task_get_requirement_f(task, idx) &
        bind(C, name="legion_task_get_requirement")
      use iso_c_binding
      import legion_region_requirement_f_t
      import legion_task_f_t
      implicit none

      type(legion_region_requirement_f_t)      :: legion_task_get_requirement_f
      type(legion_task_f_t), value, intent(in) :: task
      integer(c_int), value, intent(in)        :: idx
    end function legion_task_get_requirement_f

    ! @see Legion::Task::futures
    function legion_task_get_future_f(task, idx) &
        bind(C, name="legion_task_get_future")
      use iso_c_binding
      import legion_future_f_t
      import legion_task_f_t
      implicit none

      type(legion_future_f_t)                  :: legion_task_get_future_f
      type(legion_task_f_t), value, intent(in) :: task
      integer(c_int), value, intent(in)        :: idx
    end function legion_task_get_future_f

    ! @see Legion::Task::futures
    function legion_task_get_futures_size_f(task) &
        bind(C, name="legion_task_get_futures_size")
      use iso_c_binding
      import legion_task_f_t
      implicit none

      integer(c_int)                  :: legion_task_get_futures_size_f
      type(legion_task_f_t), value, intent(in) :: task
    end function legion_task_get_futures_size_f

    ! -----------------------------------------------------------------------
    ! Domain Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Domain::from_rect()
    function legion_domain_from_rect_1d_f(r) &
        bind(C, name="legion_domain_from_rect_1d")
      use iso_c_binding
      import legion_rect_1d_f_t
      import legion_domain_f_t
      implicit none

      type(legion_domain_f_t)                     :: legion_domain_from_rect_1d_f
      type(legion_rect_1d_f_t), value, intent(in) :: r
    end function legion_domain_from_rect_1d_f

#if LEGION_MAX_DIM >= 2
    ! @see Legion::Domain::from_rect()
    function legion_domain_from_rect_2d_f(r) &
        bind(C, name="legion_domain_from_rect_2d")
      use iso_c_binding
      import legion_rect_2d_f_t
      import legion_domain_f_t
      implicit none

      type(legion_domain_f_t)                     :: legion_domain_from_rect_2d_f
      type(legion_rect_2d_f_t), value, intent(in) :: r
    end function legion_domain_from_rect_2d_f
#endif

#if LEGION_MAX_DIM >= 3
    ! @see Legion::Domain::from_rect()
    function legion_domain_from_rect_3d_f(r) &
        bind(C, name="legion_domain_from_rect_3d")
      use iso_c_binding
      import legion_rect_3d_f_t
      import legion_domain_f_t
      implicit none

      type(legion_domain_f_t)                     :: legion_domain_from_rect_3d_f
      type(legion_rect_3d_f_t), value, intent(in) :: r
    end function legion_domain_from_rect_3d_f
#endif

    ! @see Legion::Domain::get_rect()
    function legion_domain_get_rect_1d_f(d) &
        bind(C, name="legion_domain_get_rect_1d")
      use iso_c_binding
      import legion_rect_1d_f_t
      import legion_domain_f_t
      implicit none

      type(legion_rect_1d_f_t)                   :: legion_domain_get_rect_1d_f
      type(legion_domain_f_t), value, intent(in) :: d
    end function legion_domain_get_rect_1d_f

#if LEGION_MAX_DIM >= 2
    ! @see Legion::Domain::get_rect()
    function legion_domain_get_rect_2d_f(d) &
        bind(C, name="legion_domain_get_rect_2d")
      use iso_c_binding
      import legion_rect_2d_f_t
      import legion_domain_f_t
      implicit none

      type(legion_rect_2d_f_t)                   :: legion_domain_get_rect_2d_f
      type(legion_domain_f_t), value, intent(in) :: d
    end function legion_domain_get_rect_2d_f
#endif

#if LEGION_MAX_DIM >= 3
    ! @see Legion::Domain::get_rect()
    function legion_domain_get_rect_3d_f(d) &
        bind(C, name="legion_domain_get_rect_3d")
      use iso_c_binding
      import legion_rect_3d_f_t
      import legion_domain_f_t
      implicit none

      type(legion_rect_3d_f_t)                   :: legion_domain_get_rect_3d_f
      type(legion_domain_f_t), value, intent(in) :: d
    end function legion_domain_get_rect_3d_f
#endif

    ! @see Legion::Domain::get_volume()
    function legion_domain_get_volume_f(d) &
        bind(C, name="legion_domain_get_volume")
      use iso_c_binding
      import legion_domain_f_t
      implicit none

      integer(c_size_t)                          :: legion_domain_get_volume_f
      type(legion_domain_f_t), value, intent(in) :: d
    end function legion_domain_get_volume_f

    ! -----------------------------------------------------------------------
    ! Domain Transform Operations
    ! -----------------------------------------------------------------------
    function legion_domain_transform_from_1x1_f(t) &
        bind(C, name="legion_domain_transform_from_1x1")
      use iso_c_binding
      import legion_domain_transform_f_t
      import legion_transform_1x1_f_t

      type(legion_domain_transform_f_t)                 :: legion_domain_transform_from_1x1_f
      type(legion_transform_1x1_f_t), value, intent(in) :: t
    end function legion_domain_transform_from_1x1_f

#if LEGION_MAX_DIM >= 2
    function legion_domain_transform_from_2x2_f(t) &
        bind(C, name="legion_domain_transform_from_2x2")
      use iso_c_binding
      import legion_domain_transform_f_t
      import legion_transform_2x2_f_t

      type(legion_domain_transform_f_t)                 :: legion_domain_transform_from_2x2_f
      type(legion_transform_2x2_f_t), value, intent(in) :: t
    end function legion_domain_transform_from_2x2_f
#endif

#if LEGION_MAX_DIM >= 3
    function legion_domain_transform_from_3x3_f(t) &
        bind(C, name="legion_domain_transform_from_3x3")
      use iso_c_binding
      import legion_domain_transform_f_t
      import legion_transform_3x3_f_t

      type(legion_domain_transform_f_t)                 :: legion_domain_transform_from_3x3_f
      type(legion_transform_3x3_f_t), value, intent(in) :: t
    end function legion_domain_transform_from_3x3_f
#endif

    ! -----------------------------------------------------------------------
    ! Domain Point Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Domain::from_point()
    function legion_domain_point_from_point_1d_f(p) &
        bind(C, name="legion_domain_point_from_point_1d")
      use iso_c_binding
      import legion_domain_point_f_t
      import legion_point_1d_f_t
      implicit none

      type(legion_domain_point_f_t)                :: legion_domain_point_from_point_1d_f
      type(legion_point_1d_f_t), value, intent(in) :: p
    end function legion_domain_point_from_point_1d_f

#if LEGION_MAX_DIM >= 2
    ! @see Legion::Domain::from_point()
    function legion_domain_point_from_point_2d_f(p) &
        bind(C, name="legion_domain_point_from_point_2d")
      use iso_c_binding
      import legion_domain_point_f_t
      import legion_point_2d_f_t
      implicit none

      type(legion_domain_point_f_t)                :: legion_domain_point_from_point_2d_f
      type(legion_point_2d_f_t), value, intent(in) :: p
    end function legion_domain_point_from_point_2d_f
#endif

#if LEGION_MAX_DIM >= 3
    ! @see Legion::Domain::from_point()
    function legion_domain_point_from_point_3d_f(p) &
        bind(C, name="legion_domain_point_from_point_3d")
      use iso_c_binding
      import legion_domain_point_f_t
      import legion_point_3d_f_t
      implicit none

      type(legion_domain_point_f_t)                :: legion_domain_point_from_point_3d_f
      type(legion_point_3d_f_t), value, intent(in) :: p
    end function legion_domain_point_from_point_3d_f
#endif

    ! @see Legion::DomainPoint::get_point()
    function legion_domain_point_get_point_1d_f(p) &
        bind(C, name="legion_domain_point_get_point_1d")
      use iso_c_binding
      import legion_domain_point_f_t
      import legion_point_1d_f_t
      implicit none

      type(legion_point_1d_f_t)                        :: legion_domain_point_get_point_1d_f
      type(legion_domain_point_f_t), value, intent(in) :: p
    end function legion_domain_point_get_point_1d_f

#if LEGION_MAX_DIM >= 2
    ! @see Legion::DomainPoint::get_point()
    function legion_domain_point_get_point_2d_f(p) &
        bind(C, name="legion_domain_point_get_point_2d")
      use iso_c_binding
      import legion_domain_point_f_t
      import legion_point_2d_f_t
      implicit none

      type(legion_point_2d_f_t)                        :: legion_domain_point_get_point_2d_f
      type(legion_domain_point_f_t), value, intent(in) :: p
    end function legion_domain_point_get_point_2d_f
#endif

#if LEGION_MAX_DIM >= 3
    ! @see Legion::DomainPoint::get_point()
    function legion_domain_point_get_point_3d_f(p) &
        bind(C, name="legion_domain_point_get_point_3d")
      use iso_c_binding
      import legion_domain_point_f_t
      import legion_point_3d_f_t

      type(legion_point_3d_f_t)                        :: legion_domain_point_get_point_3d_f
      type(legion_domain_point_f_t), value, intent(in) :: p
    end function legion_domain_point_get_point_3d_f
#endif

    ! -----------------------------------------------------------------------
    ! Domain Point Iterator
    ! -----------------------------------------------------------------------
    ! @see Legion::Domain::DomainPointIterator::DomainPointIterator()
    function legion_domain_point_iterator_create_f(handle) &
        bind(C, name="legion_domain_point_iterator_create")
      use iso_c_binding
      import legion_domain_point_iterator_f_t
      import legion_domain_f_t
      implicit none

      type(legion_domain_point_iterator_f_t)     :: legion_domain_point_iterator_create_f
      type(legion_domain_f_t), value, intent(in) :: handle
    end function legion_domain_point_iterator_create_f

    ! @see Legion::Domain::DomainPointIterator::~DomainPointIterator()
    subroutine legion_domain_point_iterator_destroy_f(handle) &
        bind(C, name="legion_domain_point_iterator_destroy")
      use iso_c_binding
      import legion_domain_point_iterator_f_t

      type(legion_domain_point_iterator_f_t), value, intent(in) :: handle
    end subroutine legion_domain_point_iterator_destroy_f

    ! @see Legion::Domain::DomainPointIterator::any_left
    function legion_domain_point_iterator_has_next_f(handle) &
        bind(C, name="legion_domain_point_iterator_has_next")
      use iso_c_binding
      import legion_domain_point_iterator_f_t

      logical(c_bool)                                           :: legion_domain_point_iterator_has_next_f
      type(legion_domain_point_iterator_f_t), value, intent(in) :: handle
    end function legion_domain_point_iterator_has_next_f

    ! @see Legion::Domain::DomainPointIterator::step()
    function legion_domain_point_iterator_next_f(handle) &
        bind(C, name="legion_domain_point_iterator_next")
      use iso_c_binding
      import legion_domain_point_f_t
      import legion_domain_point_iterator_f_t

      type(legion_domain_point_f_t)                             :: legion_domain_point_iterator_next_f
      type(legion_domain_point_iterator_f_t), value, intent(in) :: handle
    end function legion_domain_point_iterator_next_f

    ! -----------------------------------------------------------------------
    ! Future Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Future::Future()
    function legion_future_copy_f(handle) &
        bind(C, name="legion_future_copy")
      use iso_c_binding
      import legion_future_f_t
      implicit none

      type(legion_future_f_t)                    :: legion_future_copy_f
      type(legion_future_f_t), value, intent(in) :: handle
    end function legion_future_copy_f

    ! @see Legion::Future::~Future()
    subroutine legion_future_destroy_f(handle) &
        bind(C, name="legion_future_destroy")
      use iso_c_binding
      import legion_future_f_t
      implicit none

      type(legion_future_f_t), value, intent(in) :: handle
    end subroutine legion_future_destroy_f

    ! @see Legion::Future::is_ready()
    function legion_future_is_ready_f(handle) &
        bind(C, name="legion_future_is_ready")
      use iso_c_binding
      import legion_future_f_t
      implicit none

      logical(c_bool)                            :: legion_future_is_ready_f
      type(legion_future_f_t), value, intent(in) :: handle
    end function legion_future_is_ready_f

    ! @see Legion::Future::get_untyped_pointer()
    function legion_future_get_untyped_pointer_f(handle) &
        bind(C, name="legion_future_get_untyped_pointer")
      use iso_c_binding
      import legion_future_f_t
      implicit none

      type(c_ptr)                                :: legion_future_get_untyped_pointer_f
      type(legion_future_f_t), value, intent(in) :: handle
    end function legion_future_get_untyped_pointer_f

    ! @see Legion::Future::get_untyped_size()
    function legion_future_get_untyped_size_f(handle) &
        bind(C, name="legion_future_get_untyped_size")
      use iso_c_binding
      import legion_future_f_t
      implicit none

      integer(c_size_t)                          :: legion_future_get_untyped_size_f
      type(legion_future_f_t), value, intent(in) :: handle
    end function legion_future_get_untyped_size_f

    ! -----------------------------------------------------------------------
    ! Future Map Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::FutureMap::wait_all_results()
    subroutine legion_future_map_wait_all_results_f(handle) &
        bind(C, name="legion_future_map_wait_all_results")
      use iso_c_binding
      import legion_future_map_f_t
      implicit none

      type(legion_future_map_f_t), value, intent(in) :: handle
    end subroutine legion_future_map_wait_all_results_f

    ! @see Legion::FutureMap::wait_all_results()
    function legion_future_map_get_future_f(handle, point) &
        bind(C, name="legion_future_map_get_future")
      use iso_c_binding
      import legion_future_f_t
      import legion_future_map_f_t
      import legion_domain_point_f_t
      implicit none

      type(legion_future_f_t)                          :: legion_future_map_get_future_f
      type(legion_future_map_f_t), value, intent(in)   :: handle
      type(legion_domain_point_f_t), value, intent(in) :: point
    end function legion_future_map_get_future_f

    ! -----------------------------------------------------------------------
    ! Index Space Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_index_space(Context, size_t)
    function legion_index_space_create_f(runtime, ctx, max_num_elmts) &
        bind(C, name="legion_index_space_create")
      use iso_c_binding
      import legion_index_space_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      implicit none

      type(legion_index_space_f_t)                :: legion_index_space_create_f
      type(legion_runtime_f_t), value, intent(in) :: runtime
      type(legion_context_f_t), value, intent(in) :: ctx
      integer(c_size_t), value, intent(in)        :: max_num_elmts
    end function legion_index_space_create_f

    ! @see Legion::Runtime::create_index_space(Context, Domain)
    function legion_index_space_create_domain_f(runtime, ctx, domain) &
        bind(C, name="legion_index_space_create_domain")
      use iso_c_binding
      import legion_index_space_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_domain_f_t
      implicit none

      type(legion_index_space_f_t)                :: legion_index_space_create_domain_f
      type(legion_runtime_f_t), value, intent(in) :: runtime
      type(legion_context_f_t), value, intent(in) :: ctx
      type(legion_domain_f_t), value, intent(in)  :: domain
    end function legion_index_space_create_domain_f

    ! @see Legion::Runtime::destroy_index_space()
    subroutine legion_index_space_destroy_f(runtime, ctx, handle) &
        bind(C, name="legion_index_space_destroy")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)      :: runtime
      type(legion_context_f_t), value, intent(in)      :: ctx
      type(legion_index_space_f_t), value, intent(in)  :: handle
    end subroutine legion_index_space_destroy_f

    ! @see Legion::Runtime::get_index_space_domain()
    function legion_index_space_get_domain_f(runtime, handle) &
        bind(C, name="legion_index_space_get_domain")
      use iso_c_binding
      import legion_domain_f_t
      import legion_runtime_f_t
      import legion_index_space_f_t
      implicit none

      type(legion_domain_f_t)                         :: legion_index_space_get_domain_f
      type(legion_runtime_f_t), value, intent(in)     :: runtime
      type(legion_index_space_f_t), value, intent(in) :: handle
    end function legion_index_space_get_domain_f

    ! @see Legion::Runtime::attach_name()
    subroutine legion_index_space_attach_name_f(runtime, handle, name, is_mutable) &
        bind (C, name="legion_index_space_attach_name")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_index_space_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)     :: runtime
      type(legion_index_space_f_t), value, intent(in) :: handle
      type(c_ptr), value, intent(in)                 :: name
      logical(c_bool), value, intent(in)              :: is_mutable
    end subroutine legion_index_space_attach_name_f

    ! -----------------------------------------------------------------------
    ! Index Partition Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_equal_partition()
    function legion_index_partition_create_equal_f(runtime, ctx, parent, &
        color_space, granularity, color) &
        bind(C, name="legion_index_partition_create_equal")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      implicit none

      type(legion_index_partition_f_t)                :: legion_index_partition_create_equal_f
      type(legion_runtime_f_t), value, intent(in)     :: runtime
      type(legion_context_f_t), value, intent(in)     :: ctx
      type(legion_index_space_f_t), value, intent(in) :: parent
      type(legion_index_space_f_t), value, intent(in) :: color_space
      integer(c_size_t), value, intent(in)            :: granularity
      integer(c_int), value, intent(in)               :: color
    end function legion_index_partition_create_equal_f

    ! @see Legion::Runtime::create_partition_by_union()
    function legion_index_partition_create_by_union_f(runtime, ctx, parent, &
        handle1, handle2, color_space, part_kind, color) &
        bind(C, name="legion_index_partition_create_by_union")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      implicit none

      type(legion_index_partition_f_t)                    :: legion_index_partition_create_by_union_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_index_space_f_t), value, intent(in)     :: parent
      type(legion_index_partition_f_t), value, intent(in) :: handle1
      type(legion_index_partition_f_t), value, intent(in) :: handle2
      type(legion_index_space_f_t), value, intent(in)     :: color_space
      integer(c_int), value, intent(in)                   :: part_kind
      integer(c_int), value, intent(in)                   :: color
    end function legion_index_partition_create_by_union_f

    ! @see Legion::Runtime::create_partition_by_intersection()
    function legion_index_partition_create_by_intersection_f(runtime, ctx, parent, &
        handle1, handle2, color_space, part_kind, color) &
        bind(C, name="legion_index_partition_create_by_intersection")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      implicit none

      type(legion_index_partition_f_t)                    :: legion_index_partition_create_by_intersection_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_index_space_f_t), value, intent(in)     :: parent
      type(legion_index_partition_f_t), value, intent(in) :: handle1
      type(legion_index_partition_f_t), value, intent(in) :: handle2
      type(legion_index_space_f_t), value, intent(in)     :: color_space
      integer(c_int), value, intent(in)                   :: part_kind
      integer(c_int), value, intent(in)                   :: color
    end function legion_index_partition_create_by_intersection_f

    !
    function legion_index_partition_create_by_intersection_mirror_f(runtime, ctx, parent, &
        handle, part_kind, color, dominates) &
        bind(C, name="legion_index_partition_create_by_intersection_mirror")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      implicit none

      type(legion_index_partition_f_t)                    :: legion_index_partition_create_by_intersection_mirror_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_index_space_f_t), value, intent(in)     :: parent
      type(legion_index_partition_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                   :: part_kind
      integer(c_int), value, intent(in)                   :: color
      logical(c_bool), value, intent(in)                  :: dominates
    end function legion_index_partition_create_by_intersection_mirror_f

    ! @see Legion::Runtime::create_partition_by_difference()
    function legion_index_partition_create_by_difference_f(runtime, ctx, parent, &
        handle1, handle2, color_space, part_kind, color) &
        bind(C, name="legion_index_partition_create_by_difference")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      implicit none

      type(legion_index_partition_f_t)                    :: legion_index_partition_create_by_difference_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_index_space_f_t), value, intent(in)     :: parent
      type(legion_index_partition_f_t), value, intent(in) :: handle1
      type(legion_index_partition_f_t), value, intent(in) :: handle2
      type(legion_index_space_f_t), value, intent(in)     :: color_space
      integer(c_int), value, intent(in)                   :: part_kind
      integer(c_int), value, intent(in)                   :: color
    end function legion_index_partition_create_by_difference_f

    ! @see Legion::Runtime::create_partition_by_image()
    function legion_index_partition_create_by_image_f(runtime, ctx, handle, &
        projection, parent, fid, color_space, part_kind, color) &
        bind(C, name="legion_index_partition_create_by_image")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      import legion_logical_partition_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_index_partition_f_t)                      :: legion_index_partition_create_by_image_f
      type(legion_runtime_f_t), value, intent(in)           :: runtime
      type(legion_context_f_t), value, intent(in)           :: ctx
      type(legion_index_space_f_t), value, intent(in)       :: handle
      type(legion_logical_partition_f_t), value, intent(in) :: projection
      type(legion_logical_region_f_t), value, intent(in)    :: parent
      integer(c_int), value, intent(in)                     :: fid
      type(legion_index_space_f_t), value, intent(in)       :: color_space
      integer(c_int), value, intent(in)                     :: part_kind
      integer(c_int), value, intent(in)                     :: color
    end function legion_index_partition_create_by_image_f

    ! @see Legion::Runtime::create_partition_by_preimage()
    function legion_index_partition_create_by_preimage_f(runtime, ctx, projection, &
        handle, parent, fid, color_space, part_kind, color) &
        bind(C, name="legion_index_partition_create_by_preimage")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      import legion_logical_partition_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_index_partition_f_t)                      :: legion_index_partition_create_by_preimage_f
      type(legion_runtime_f_t), value, intent(in)           :: runtime
      type(legion_context_f_t), value, intent(in)           :: ctx
      type(legion_index_partition_f_t), value, intent(in)   :: projection
      type(legion_logical_region_f_t), value, intent(in)    :: handle
      type(legion_logical_region_f_t), value, intent(in)    :: parent
      integer(c_int), value, intent(in)                     :: fid
      type(legion_index_space_f_t), value, intent(in)       :: color_space
      integer(c_int), value, intent(in)                     :: part_kind
      integer(c_int), value, intent(in)                     :: color
    end function legion_index_partition_create_by_preimage_f

    ! @see Legion::Runtime::create_partition_by_image_range()
    function legion_index_partition_create_by_image_range_f(runtime, ctx, handle, &
        projection, parent, fid, color_space, part_kind, color) &
        bind(C, name="legion_index_partition_create_by_image_range")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      import legion_logical_partition_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_index_partition_f_t)                      :: legion_index_partition_create_by_image_range_f
      type(legion_runtime_f_t), value, intent(in)           :: runtime
      type(legion_context_f_t), value, intent(in)           :: ctx
      type(legion_index_space_f_t), value, intent(in)       :: handle
      type(legion_logical_partition_f_t), value, intent(in) :: projection
      type(legion_logical_region_f_t), value, intent(in)    :: parent
      integer(c_int), value, intent(in)                     :: fid
      type(legion_index_space_f_t), value, intent(in)       :: color_space
      integer(c_int), value, intent(in)                     :: part_kind
      integer(c_int), value, intent(in)                     :: color
    end function legion_index_partition_create_by_image_range_f

    ! @see Legion::Runtime::create_partition_by_preimage_range()
    function legion_index_partition_create_by_preimage_range_f(runtime, ctx, projection, &
        handle, parent, fid, color_space, part_kind, color) &
        bind(C, name="legion_index_partition_create_by_preimage_range")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      import legion_logical_partition_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_index_partition_f_t)                      :: legion_index_partition_create_by_preimage_range_f
      type(legion_runtime_f_t), value, intent(in)           :: runtime
      type(legion_context_f_t), value, intent(in)           :: ctx
      type(legion_index_partition_f_t), value, intent(in)   :: projection
      type(legion_logical_region_f_t), value, intent(in)    :: handle
      type(legion_logical_region_f_t), value, intent(in)    :: parent
      integer(c_int), value, intent(in)                     :: fid
      type(legion_index_space_f_t), value, intent(in)       :: color_space
      integer(c_int), value, intent(in)                     :: part_kind
      integer(c_int), value, intent(in)                     :: color
    end function legion_index_partition_create_by_preimage_range_f

    ! @see Legion::Runtime::create_partition_by_restriction()
    function legion_index_partition_create_by_restriction_f(runtime, ctx, parent, &
        color_space, transform, extent, part_kind, color) &
        bind(C, name="legion_index_partition_create_by_restriction")
      use iso_c_binding
      import legion_index_partition_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      import legion_domain_transform_f_t
      import legion_domain_f_t
      implicit none

      type(legion_index_partition_f_t)                     :: legion_index_partition_create_by_restriction_f
      type(legion_runtime_f_t), value, intent(in)          :: runtime
      type(legion_context_f_t), value, intent(in)          :: ctx
      type(legion_index_space_f_t), value, intent(in)      :: parent
      type(legion_index_space_f_t), value, intent(in)      :: color_space
      type(legion_domain_transform_f_t), value, intent(in) :: transform
      type(legion_domain_f_t), value, intent(in)           :: extent
      integer(c_int), value, intent(in)                    :: part_kind
      integer(c_int), value, intent(in)                    :: color
    end function legion_index_partition_create_by_restriction_f

    ! @see Legion::Runtime::attach_name()
    subroutine legion_index_partition_attach_name_f(runtime, handle, &
        name, is_mutable) &
        bind (C, name="legion_index_partition_attach_name")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_index_partition_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_index_partition_f_t), value, intent(in) :: handle
      type(c_ptr), value, intent(in)                      :: name
      logical(c_bool), value, intent(in)                  :: is_mutable
    end subroutine legion_index_partition_attach_name_f

    ! -----------------------------------------------------------------------
    ! Logical Region Tree Traversal Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::get_logical_partition()
    function legion_logical_partition_create_f(runtime, parent, handle) &
        bind (C, name="legion_logical_partition_create")
      use iso_c_binding
      import legion_logical_partition_f_t
      import legion_runtime_f_t
      import legion_logical_region_f_t
      import legion_index_partition_f_t
      implicit none

      type(legion_logical_partition_f_t)                  :: legion_logical_partition_create_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_logical_region_f_t), value, intent(in)  :: parent
      type(legion_index_partition_f_t), value, intent(in) :: handle
    end function legion_logical_partition_create_f

    ! @see Legion::Runtime::destroy_logical_partition()
    subroutine legion_logical_partition_destroy_f(runtime, ctx, handle) &
        bind (C, name="legion_logical_partition_destroy")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_logical_partition_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)           :: runtime
      type(legion_context_f_t), value, intent(in)           :: ctx
      type(legion_logical_partition_f_t), value, intent(in) :: handle
    end subroutine legion_logical_partition_destroy_f

    ! @see Legion::Runtime::attach_name()
    subroutine legion_logical_partition_attach_name_f(runtime, handle, &
        name, is_mutable) &
        bind (C, name="legion_logical_partition_attach_name")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_logical_partition_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)           :: runtime
      type(legion_logical_partition_f_t), value, intent(in) :: handle
      type(c_ptr), value, intent(in)                        :: name
      logical(c_bool), value, intent(in)                    :: is_mutable
    end subroutine legion_logical_partition_attach_name_f

    ! -----------------------------------------------------------------------
    ! Field Space Operatiins
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_field_space()
    function legion_field_space_create_f(runtime, ctx) &
        bind(C, name="legion_field_space_create")
      use iso_c_binding
      import legion_field_space_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      implicit none

      type(legion_field_space_f_t)                :: legion_field_space_create_f
      type(legion_runtime_f_t), value, intent(in) :: runtime
      type(legion_context_f_t), value, intent(in) :: ctx
    end function legion_field_space_create_f

    ! @see Legion::Runtime::destroy_field_space()
    subroutine legion_field_space_destroy_f(runtime, ctx, handle) &
        bind(C, name="legion_field_space_destroy")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_field_space_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)     :: runtime
      type(legion_context_f_t), value, intent(in)     :: ctx
      type(legion_field_space_f_t), value, intent(in) :: handle
    end subroutine legion_field_space_destroy_f

    ! -----------------------------------------------------------------------
    ! Field Allocator
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_field_allocator()
    function legion_field_allocator_create_f(runtime, ctx, handle) &
        bind(C, name="legion_field_allocator_create")
      use iso_c_binding
      import legion_field_allocator_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_field_space_f_t
      implicit none

      type(legion_field_allocator_f_t)                :: legion_field_allocator_create_f
      type(legion_runtime_f_t), value, intent(in)     :: runtime
      type(legion_context_f_t), value, intent(in)     :: ctx
      type(legion_field_space_f_t), value, intent(in) :: handle
    end function legion_field_allocator_create_f

    ! @see Legion::FieldAllocator::~FieldAllocator()
    subroutine legion_field_allocator_destroy_f(handle) &
        bind(C, name="legion_field_allocator_destroy")
      use iso_c_binding
      import legion_field_allocator_f_t
      implicit none

      type(legion_field_allocator_f_t), value, intent(in) :: handle
    end subroutine legion_field_allocator_destroy_f

    ! @see Legion::FieldAllocator::allocate_field()
    function legion_field_allocator_allocate_field_f(allocator, &
        field_size, desired_fieldid) &
        bind (C, name="legion_field_allocator_allocate_field")
      use iso_c_binding
      import legion_field_allocator_f_t
      implicit none

      integer(c_int)                                      :: legion_field_allocator_allocate_field_f
      type(legion_field_allocator_f_t), value, intent(in) :: allocator
      integer(c_size_t), value, intent(in)                :: field_size
      integer(c_int), value, intent(in)                   :: desired_fieldid
    end function legion_field_allocator_allocate_field_f

    ! @see Legion::FieldAllocator::free_field()
    subroutine legion_field_allocator_free_field_f(allocator, fid) &
        bind (C, name="legion_field_allocator_free_field")
      use iso_c_binding
      import legion_field_allocator_f_t
      implicit none

      type(legion_field_allocator_f_t), value, intent(in) :: allocator
      integer(c_int), value, intent(in)                   :: fid
    end subroutine legion_field_allocator_free_field_f

    ! -----------------------------------------------------------------------
    ! Logical Region
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_logical_region()
    function legion_logical_region_create_f(runtime, ctx, index, fields) &
        bind(C, name="legion_logical_region_create")
      use iso_c_binding
      import legion_logical_region_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_space_f_t
      import legion_field_space_f_t
      implicit none

      type(legion_logical_region_f_t)                 :: legion_logical_region_create_f
      type(legion_runtime_f_t), value, intent(in)     :: runtime
      type(legion_context_f_t), value, intent(in)     :: ctx
      type(legion_index_space_f_t), value, intent(in) :: index
      type(legion_field_space_f_t), value, intent(in) :: fields
    end function legion_logical_region_create_f

    ! @see Legion::Runtime::destroy_logical_region()
    subroutine legion_logical_region_destroy_f(runtime, ctx, handle) &
        bind(C, name="legion_logical_region_destroy")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)        :: runtime
      type(legion_context_f_t), value, intent(in)        :: ctx
      type(legion_logical_region_f_t), value, intent(in) :: handle
    end subroutine legion_logical_region_destroy_f

    ! @see Legion::LogicalRegion::get_index_space
    function legion_logical_region_get_index_space_f(handle) &
        bind(C, name="legion_logical_region_get_index_space")
      use iso_c_binding
      import legion_index_space_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_index_space_f_t)                       :: legion_logical_region_get_index_space_f
      type(legion_logical_region_f_t), value, intent(in) :: handle
    end function legion_logical_region_get_index_space_f

    ! -----------------------------------------------------------------------
    ! Region Requirement Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::RegionRequirement::region
    function legion_region_requirement_get_region_f(handle) &
        bind(C, name="legion_region_requirement_get_region")
      use iso_c_binding
      import legion_region_requirement_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_logical_region_f_t)                        :: legion_region_requirement_get_region_f
      type(legion_region_requirement_f_t), value, intent(in) :: handle
    end function legion_region_requirement_get_region_f

    ! @see Legion::RegionRequirement::privilege_fields
    function legion_region_requirement_get_privilege_field_f(handle, idx) &
        bind (C, name="legion_region_requirement_get_privilege_field")
      use iso_c_binding
      import legion_region_requirement_f_t
      implicit none

      integer(c_int)                                         :: legion_region_requirement_get_privilege_field_f
      type(legion_region_requirement_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                      :: idx
    end function legion_region_requirement_get_privilege_field_f

    ! -----------------------------------------------------------------------
    ! Physical Data Operations
    ! -----------------------------------------------------------------------
    function legion_get_physical_region_by_id_f(regionptr, id, num_regions) &
        bind(C, name="legion_get_physical_region_by_id")
      use iso_c_binding
      import legion_physical_region_f_t
      implicit none

      type(legion_physical_region_f_t)  :: legion_get_physical_region_by_id_f
      type(c_ptr), value, intent(in)    :: regionptr
      integer(c_int), value, intent(in) :: id
      integer(c_int), value, intent(in) :: num_regions
    end function legion_get_physical_region_by_id_f

    ! @see Legion::PhysicalRegion::~PhysicalRegion()
    subroutine legion_physical_region_destroy_f(handle) &
        bind(C, name="legion_physical_region_destroy")
      use iso_c_binding
      import legion_physical_region_f_t
      implicit none

      type(legion_physical_region_f_t), value, intent(in) :: handle
    end subroutine legion_physical_region_destroy_f

    ! @see Legion::PhysicalRegion::is_mapped()
    function legion_physical_region_is_mapped_f(handle) &
        bind(C, name="legion_physical_region_is_mapped")
      use iso_c_binding
      import legion_physical_region_f_t
      implicit none

      logical(c_bool)                                     :: legion_physical_region_is_mapped_f
      type(legion_physical_region_f_t), value, intent(in) :: handle
    end function legion_physical_region_is_mapped_f

    ! @see Legion::PhysicalRegion::wait_until_valid()
    subroutine legion_physical_region_wait_until_valid_f(handle) &
        bind(C, name="legion_physical_region_wait_until_valid")
      use iso_c_binding
      import legion_physical_region_f_t
      implicit none

      type(legion_physical_region_f_t), value, intent(in) :: handle
    end subroutine legion_physical_region_wait_until_valid_f

    ! @see Legion::PhysicalRegion::is_valid()
    function legion_physical_region_is_valid_f(handle) &
        bind(C, name="legion_physical_region_is_valid")
      use iso_c_binding
      import legion_physical_region_f_t
      implicit none

      logical(c_bool)                                     :: legion_physical_region_is_valid_f
      type(legion_physical_region_f_t), value, intent(in) :: handle
    end function legion_physical_region_is_valid_f

    ! @see Legion::PhysicalRegion::get_logical_region()
    function legion_physical_region_get_logical_region_f(handle) &
        bind(C, name="legion_physical_region_get_logical_region")
      use iso_c_binding
      import legion_physical_region_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_logical_region_f_t)                     :: legion_physical_region_get_logical_region_f
      type(legion_physical_region_f_t), value, intent(in) :: handle
    end function legion_physical_region_get_logical_region_f

    ! @see Legion::PhysicalRegion::get_field_accessor()
    function legion_physical_region_get_field_accessor_array_1d_f(handle, fid) &
        bind(C, name="legion_physical_region_get_field_accessor_array_1d")
      use iso_c_binding
      import legion_accessor_array_1d_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_accessor_array_1d_f_t)                  :: legion_physical_region_get_field_accessor_array_1d_f
      type(legion_physical_region_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                   :: fid
    end function legion_physical_region_get_field_accessor_array_1d_f

#if LEGION_MAX_DIM >= 2
    ! @see Legion::PhysicalRegion::get_field_accessor()
    function legion_physical_region_get_field_accessor_array_2d_f(handle, fid) &
        bind(C, name="legion_physical_region_get_field_accessor_array_2d")
      use iso_c_binding
      import legion_accessor_array_2d_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_accessor_array_2d_f_t)                  :: legion_physical_region_get_field_accessor_array_2d_f
      type(legion_physical_region_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                   :: fid
    end function legion_physical_region_get_field_accessor_array_2d_f
#endif

#if LEGION_MAX_DIM >= 3
    ! @see Legion::PhysicalRegion::get_field_accessor()
    function legion_physical_region_get_field_accessor_array_3d_f(handle, fid) &
        bind(C, name="legion_physical_region_get_field_accessor_array_3d")
      use iso_c_binding
      import legion_accessor_array_3d_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_accessor_array_3d_f_t)                  :: legion_physical_region_get_field_accessor_array_3d_f
      type(legion_physical_region_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                   :: fid
    end function legion_physical_region_get_field_accessor_array_3d_f
#endif

    ! @see Legion::UnsafeFieldAccessor::ptr
    function legion_accessor_array_1d_raw_rect_ptr_f(handle, rect, subrect, offset) &
        bind(C, name="legion_accessor_array_1d_raw_rect_ptr")
      use iso_c_binding
      import legion_accessor_array_1d_f_t
      import legion_rect_1d_f_t
      import legion_byte_offset_f_t
      implicit none

      type(c_ptr)                                           :: legion_accessor_array_1d_raw_rect_ptr_f
      type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
      type(legion_rect_1d_f_t), value, intent(in)           :: rect
      type(legion_rect_1d_f_t), intent(out)                 :: subrect ! pass reference
      type(legion_byte_offset_f_t), intent(out)             :: offset  ! pass reference
    end function legion_accessor_array_1d_raw_rect_ptr_f

#if LEGION_MAX_DIM >= 2
    ! @see Legion::UnsafeFieldAccessor::ptr
    function legion_accessor_array_2d_raw_rect_ptr_f(handle, rect, subrect, offset) &
        bind(C, name="legion_accessor_array_2d_raw_rect_ptr")
      use iso_c_binding
      import legion_accessor_array_2d_f_t
      import legion_rect_2d_f_t
      import legion_byte_offset_f_t
      implicit none

      type(c_ptr)                                           :: legion_accessor_array_2d_raw_rect_ptr_f
      type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
      type(legion_rect_2d_f_t), value, intent(in)           :: rect
      type(legion_rect_2d_f_t), intent(out)                 :: subrect ! pass reference
      type(legion_byte_offset_f_t), intent(out)             :: offset(2)  ! pass reference
    end function legion_accessor_array_2d_raw_rect_ptr_f
#endif

#if LEGION_MAX_DIM >= 3
    ! @see Legion::UnsafeFieldAccessor::ptr
    function legion_accessor_array_3d_raw_rect_ptr_f(handle, rect, subrect, offset) &
        bind(C, name="legion_accessor_array_3d_raw_rect_ptr")
      use iso_c_binding
      import legion_accessor_array_3d_f_t
      import legion_rect_3d_f_t
      import legion_byte_offset_f_t
      implicit none

      type(c_ptr)                                           :: legion_accessor_array_3d_raw_rect_ptr_f
      type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
      type(legion_rect_3d_f_t), value, intent(in)           :: rect
      type(legion_rect_3d_f_t), intent(out)                 :: subrect ! pass reference
      type(legion_byte_offset_f_t), intent(out)             :: offset(3)  ! pass reference
    end function legion_accessor_array_3d_raw_rect_ptr_f
#endif

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_f(handle, point, dst, bytes) &
        bind(C, name="legion_accessor_array_1d_read_point")
      use iso_c_binding
      import legion_accessor_array_1d_f_t
      import legion_point_1d_f_t
      implicit none

      type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
      type(legion_point_1d_f_t), value, intent(in)          :: point
      type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
      integer(c_size_t), value, intent(in)                  :: bytes
    end subroutine legion_accessor_array_1d_read_point_f

#if LEGION_MAX_DIM >= 2
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_2d_read_point_f(handle, point, dst, bytes) &
        bind(C, name="legion_accessor_array_2d_read_point")
      use iso_c_binding
      import legion_accessor_array_2d_f_t
      import legion_point_2d_f_t
      implicit none

      type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
      type(legion_point_2d_f_t), value, intent(in)          :: point
      type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
      integer(c_size_t), value, intent(in)                  :: bytes
    end subroutine legion_accessor_array_2d_read_point_f
#endif

#if LEGION_MAX_DIM >= 3
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_3d_read_point_f(handle, point, dst, bytes) &
        bind(C, name="legion_accessor_array_3d_read_point")
      use iso_c_binding
      import legion_accessor_array_3d_f_t
      import legion_point_3d_f_t
      implicit none

      type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
      type(legion_point_3d_f_t), value, intent(in)          :: point
      type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
      integer(c_size_t), value, intent(in)                  :: bytes
    end subroutine legion_accessor_array_3d_read_point_f
#endif

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_write_point_f(handle, point, src, bytes) &
        bind(C, name="legion_accessor_array_1d_write_point")
      use iso_c_binding
      import legion_accessor_array_1d_f_t
      import legion_point_1d_f_t
      implicit none

      type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
      type(legion_point_1d_f_t), value, intent(in)          :: point
      type(c_ptr), value, intent(in)                        :: src
      integer(c_size_t), value, intent(in)                  :: bytes
    end subroutine legion_accessor_array_1d_write_point_f

#if LEGION_MAX_DIM >= 2
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_2d_write_point_f(handle, point, src, bytes) &
        bind(C, name="legion_accessor_array_2d_write_point")
      use iso_c_binding
      import legion_accessor_array_2d_f_t
      import legion_point_2d_f_t
      implicit none

      type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
      type(legion_point_2d_f_t), value, intent(in)          :: point
      type(c_ptr), value, intent(in)                        :: src
      integer(c_size_t), value, intent(in)                  :: bytes
    end subroutine legion_accessor_array_2d_write_point_f
#endif

#if LEGION_MAX_DIM >= 3
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_3d_write_point_f(handle, point, src, bytes) &
        bind(C, name="legion_accessor_array_3d_write_point")
      use iso_c_binding
      import legion_accessor_array_3d_f_t
      import legion_point_3d_f_t
      implicit none

      type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
      type(legion_point_3d_f_t), value, intent(in)          :: point
      type(c_ptr), value, intent(in)                        :: src
      integer(c_size_t), value, intent(in)                  :: bytes
    end subroutine legion_accessor_array_3d_write_point_f
#endif

    ! -----------------------------------------------------------------------
    ! Fill Field Operations
    ! -----------------------------------------------------------------------
    subroutine legion_runtime_fill_field_f(runtime, ctx, handle, parent, &
        fid, value, value_size, pred) &
        bind(C, name="legion_runtime_fill_field")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_logical_region_f_t
      import legion_predicate_f_t

      type(legion_runtime_f_t), value, intent(in)        :: runtime
      type(legion_context_f_t), value, intent(in)        :: ctx
      type(legion_logical_region_f_t), value, intent(in) :: handle
      type(legion_logical_region_f_t), value, intent(in) :: parent
      integer(c_int), value, intent(in)                  :: fid
      type(c_ptr), value, intent(in)                     :: value
      integer(c_size_t), value, intent(in)               :: value_size
      type(legion_predicate_f_t), value, intent(in)      :: pred
    end subroutine legion_runtime_fill_field_f

    ! -----------------------------------------------------------------------
    ! File Operations
    ! -----------------------------------------------------------------------
    function legion_field_map_create_f() &
        bind(C, name="legion_field_map_create")
      use iso_c_binding
      import legion_field_map_f_t
      implicit none

      type(legion_field_map_f_t) :: legion_field_map_create_f
    end function legion_field_map_create_f

    subroutine legion_field_map_destroy_f(handle) &
        bind(C, name="legion_field_map_destroy")
      use iso_c_binding
      import legion_field_map_f_t
      implicit none

      type(legion_field_map_f_t), value, intent(in) :: handle
    end subroutine legion_field_map_destroy_f

    subroutine legion_field_map_insert_f(handle, key, value) &
        bind(C, name="legion_field_map_insert")
      use iso_c_binding
      import legion_field_map_f_t
      implicit none

      type(legion_field_map_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)             :: key
      type(c_ptr), value, intent(in)                :: value
    end subroutine legion_field_map_insert_f

    ! @see Legion::Runtime::attach_hdf5()
    function legion_runtime_attach_hdf5_f(runtime, ctx, filename, &
        handle, parent, field_map, mode) &
        bind(C, name="legion_runtime_attach_hdf5")
      use iso_c_binding
      import legion_physical_region_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_logical_region_f_t
      import legion_field_map_f_t
      implicit none

      type(legion_physical_region_f_t)                   :: legion_runtime_attach_hdf5_f
      type(legion_runtime_f_t), value, intent(in)        :: runtime
      type(legion_context_f_t), value, intent(in)        :: ctx
      type(c_ptr), value, intent(in)                     :: filename
      type(legion_logical_region_f_t), value, intent(in) :: handle
      type(legion_logical_region_f_t), value, intent(in) :: parent
      type(legion_field_map_f_t), value, intent(in)      :: field_map
      integer(c_int), value, intent(in)                  :: mode
    end function legion_runtime_attach_hdf5_f

    ! @see Legion::Runtime::detach_hdf5()
    subroutine legion_runtime_detach_hdf5_f(runtime, ctx, region) &
        bind(C, name="legion_runtime_detach_hdf5")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_physical_region_f_t), value, intent(in) :: region
    end subroutine legion_runtime_detach_hdf5_f

    ! -----------------------------------------------------------------------
    ! Copy Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::CopyLauncher::CopyLauncher()
    function legion_copy_launcher_create_f(pred, id, launcher_tag) &
        bind(C, name="legion_copy_launcher_create")
      use iso_c_binding
      import legion_copy_launcher_f_t
      import legion_predicate_f_t
      implicit none

      type(legion_copy_launcher_f_t)                :: legion_copy_launcher_create_f
      type(legion_predicate_f_t), value, intent(in) :: pred
      integer(c_int), value, intent(in)             :: id
      integer(c_long), value, intent(in)            :: launcher_tag
    end function legion_copy_launcher_create_f

    ! @see Legion::CopyLauncher::~CopyLauncher()
    subroutine legion_copy_launcher_destroy_f(handle) &
        bind(C, name="legion_copy_launcher_destroy")
      use iso_c_binding
      import legion_copy_launcher_f_t
      implicit none

      type(legion_copy_launcher_f_t), value, intent(in) :: handle
    end subroutine legion_copy_launcher_destroy_f

    ! @see Legion::Runtime::issue_copy_operation()
    subroutine legion_copy_launcher_execute_f(runtime, ctx, launcher) &
        bind(C, name="legion_copy_launcher_execute")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_copy_launcher_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)       :: runtime
      type(legion_context_f_t), value, intent(in)       :: ctx
      type(legion_copy_launcher_f_t), value, intent(in) :: launcher
    end subroutine legion_copy_launcher_execute_f

    ! @see Legion::CopyLauncher::add_copy_requirements()
    function legion_copy_launcher_add_src_region_requirement_lr_f(launcher, &
        handle, priv, prop, parent, tag, verified) &
        bind(C, name="legion_copy_launcher_add_src_region_requirement_logical_region")
      use iso_c_binding
      import legion_copy_launcher_f_t
      import legion_logical_region_f_t
      implicit none

      integer(c_int)                                     :: legion_copy_launcher_add_src_region_requirement_lr_f
      type(legion_copy_launcher_f_t), value, intent(in)  :: launcher
      type(legion_logical_region_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                  :: priv
      integer(c_int), value, intent(in)                  :: prop
      type(legion_logical_region_f_t), value, intent(in) :: parent
      integer(c_long), value, intent(in)                 :: tag
      logical(c_bool), value, intent(in)                 :: verified
    end function legion_copy_launcher_add_src_region_requirement_lr_f

    ! @see Legion::CopyLauncher::add_copy_requirements()
    function legion_copy_launcher_add_dst_region_requirement_lr_f(launcher, &
        handle, priv, prop, parent, tag, verified) &
        bind(C, name="legion_copy_launcher_add_dst_region_requirement_logical_region")
      use iso_c_binding
      import legion_copy_launcher_f_t
      import legion_logical_region_f_t
      implicit none

      integer(c_int)                                     :: legion_copy_launcher_add_dst_region_requirement_lr_f
      type(legion_copy_launcher_f_t), value, intent(in)  :: launcher
      type(legion_logical_region_f_t), value, intent(in) :: handle
      integer(c_int), value, intent(in)                  :: priv
      integer(c_int), value, intent(in)                  :: prop
      type(legion_logical_region_f_t), value, intent(in) :: parent
      integer(c_long), value, intent(in)                 :: tag
      logical(c_bool), value, intent(in)                 :: verified
    end function legion_copy_launcher_add_dst_region_requirement_lr_f

    ! @see Legion::CopyLauncher::add_src_field()
    subroutine legion_copy_launcher_add_src_field_f(launcher, idx, fid, inst) &
        bind(C, name="legion_copy_launcher_add_src_field")
      use iso_c_binding
      import legion_copy_launcher_f_t
      implicit none

      type(legion_copy_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                 :: idx
      integer(c_int), value, intent(in)                 :: fid
      logical(c_bool), value, intent(in)                :: inst
    end subroutine legion_copy_launcher_add_src_field_f

    ! @see Legion::CopyLauncher::add_dst_field()
    subroutine legion_copy_launcher_add_dst_field_f(launcher, idx, fid, inst) &
        bind(C, name="legion_copy_launcher_add_dst_field")
      use iso_c_binding
      import legion_copy_launcher_f_t
      implicit none

      type(legion_copy_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                 :: idx
      integer(c_int), value, intent(in)                 :: fid
      logical(c_bool), value, intent(in)                :: inst
    end subroutine legion_copy_launcher_add_dst_field_f

    ! -----------------------------------------------------------------------
    ! Index Copy Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::IndexCopyLauncher::IndexCopyLauncher()
    function legion_index_copy_launcher_create_f(domain, pred, id, launcher_tag) &
        bind(C, name="legion_index_copy_launcher_create")
      use iso_c_binding
      import legion_index_copy_launcher_f_t
      import legion_domain_f_t
      import legion_predicate_f_t
      implicit none

      type(legion_index_copy_launcher_f_t)          :: legion_index_copy_launcher_create_f
      type(legion_domain_f_t), value, intent(in)    :: domain
      type(legion_predicate_f_t), value, intent(in) :: pred
      integer(c_int), value, intent(in)             :: id
      integer(c_long), value, intent(in)            :: launcher_tag
    end function legion_index_copy_launcher_create_f

    ! @see Legion::IndexCopyLauncher::~IndexCopyLauncher()
    subroutine legion_index_copy_launcher_destroy_f(handle) &
        bind(C, name="legion_index_copy_launcher_destroy")
      use iso_c_binding
      import legion_index_copy_launcher_f_t
      implicit none

      type(legion_index_copy_launcher_f_t), value, intent(in) :: handle
    end subroutine legion_index_copy_launcher_destroy_f

    ! @see Legion::Runtime::issue_copy_operation()
    subroutine legion_index_copy_launcher_execute_f(runtime, ctx, launcher) &
        bind(C, name="legion_index_copy_launcher_execute")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_index_copy_launcher_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in)             :: runtime
      type(legion_context_f_t), value, intent(in)             :: ctx
      type(legion_index_copy_launcher_f_t), value, intent(in) :: launcher
    end subroutine legion_index_copy_launcher_execute_f

    ! @see Legion::IndexCopyLauncher::add_copy_requirements()
    function legion_index_copy_launcher_add_src_region_requirement_lr_f(launcher, &
        handle, proj, priv, prop, parent, tag, verified) &
        bind(C, name="legion_index_copy_launcher_add_src_region_requirement_logical_region")
      use iso_c_binding
      import legion_index_copy_launcher_f_t
      import legion_logical_region_f_t
      implicit none

      integer(c_int)                                          :: legion_index_copy_launcher_add_src_region_requirement_lr_f
      type(legion_index_copy_launcher_f_t), value, intent(in) :: launcher
      type(legion_logical_region_f_t), value, intent(in)      :: handle
      integer(c_int), value, intent(in)                       :: proj
      integer(c_int), value, intent(in)                       :: priv
      integer(c_int), value, intent(in)                       :: prop
      type(legion_logical_region_f_t), value, intent(in)      :: parent
      integer(c_long), value, intent(in)                      :: tag
      logical(c_bool), value, intent(in)                      :: verified
    end function legion_index_copy_launcher_add_src_region_requirement_lr_f

    ! @see Legion::IndexCopyLauncher::add_copy_requirements()
    function legion_index_copy_launcher_add_dst_region_requirement_lr_f(launcher, &
        handle, proj, priv, prop, parent, tag, verified) &
        bind(C, name="legion_index_copy_launcher_add_dst_region_requirement_logical_region")
      use iso_c_binding
      import legion_index_copy_launcher_f_t
      import legion_logical_region_f_t
      implicit none

      integer(c_int)                                          :: legion_index_copy_launcher_add_dst_region_requirement_lr_f
      type(legion_index_copy_launcher_f_t), value, intent(in) :: launcher
      type(legion_logical_region_f_t), value, intent(in)      :: handle
      integer(c_int), value, intent(in)                       :: proj
      integer(c_int), value, intent(in)                       :: priv
      integer(c_int), value, intent(in)                       :: prop
      type(legion_logical_region_f_t), value, intent(in)      :: parent
      integer(c_long), value, intent(in)                      :: tag
      logical(c_bool), value, intent(in)                      :: verified
    end function legion_index_copy_launcher_add_dst_region_requirement_lr_f

    ! @see Legion::IndexCopyLauncher::add_copy_requirements()
    function legion_index_copy_launcher_add_src_region_requirement_lp_f(launcher, &
        handle, proj, priv, prop, parent, tag, verified) &
        bind(C, name="legion_index_copy_launcher_add_src_region_requirement_logical_partition")
      use iso_c_binding
      import legion_index_copy_launcher_f_t
      import legion_logical_partition_f_t
      import legion_logical_region_f_t
      implicit none

      integer(c_int)                                          :: legion_index_copy_launcher_add_src_region_requirement_lp_f
      type(legion_index_copy_launcher_f_t), value, intent(in) :: launcher
      type(legion_logical_partition_f_t), value, intent(in)   :: handle
      integer(c_int), value, intent(in)                       :: proj
      integer(c_int), value, intent(in)                       :: priv
      integer(c_int), value, intent(in)                       :: prop
      type(legion_logical_region_f_t), value, intent(in)      :: parent
      integer(c_long), value, intent(in)                      :: tag
      logical(c_bool), value, intent(in)                      :: verified
    end function legion_index_copy_launcher_add_src_region_requirement_lp_f

    ! @see Legion::IndexCopyLauncher::add_copy_requirements()
    function legion_index_copy_launcher_add_dst_region_requirement_lp_f(launcher, &
        handle, proj, priv, prop, parent, tag, verified) &
        bind(C, name="legion_index_copy_launcher_add_dst_region_requirement_logical_partition")
      use iso_c_binding
      import legion_index_copy_launcher_f_t
      import legion_logical_partition_f_t
      import legion_logical_region_f_t
      implicit none

      integer(c_int)                                          :: legion_index_copy_launcher_add_dst_region_requirement_lp_f
      type(legion_index_copy_launcher_f_t), value, intent(in) :: launcher
      type(legion_logical_partition_f_t), value, intent(in)   :: handle
      integer(c_int), value, intent(in)                       :: proj
      integer(c_int), value, intent(in)                       :: priv
      integer(c_int), value, intent(in)                       :: prop
      type(legion_logical_region_f_t), value, intent(in)      :: parent
      integer(c_long), value, intent(in)                      :: tag
      logical(c_bool), value, intent(in)                      :: verified
    end function legion_index_copy_launcher_add_dst_region_requirement_lp_f

    ! @see Legion::IndexCopyLauncher::add_src_field()
    subroutine legion_index_copy_launcher_add_src_field_f(launcher, idx, fid, inst) &
        bind(C, name="legion_index_copy_launcher_add_src_field")
      use iso_c_binding
      import legion_index_copy_launcher_f_t
      implicit none

      type(legion_index_copy_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                       :: idx
      integer(c_int), value, intent(in)                       :: fid
      logical(c_bool), value, intent(in)                      :: inst
    end subroutine legion_index_copy_launcher_add_src_field_f

    ! @see Legion::IndexCopyLauncher::add_dst_field()
    subroutine legion_index_copy_launcher_add_dst_field_f(launcher, idx, fid, inst) &
        bind(C, name="legion_index_copy_launcher_add_dst_field")
      use iso_c_binding
      import legion_index_copy_launcher_f_t
      implicit none

      type(legion_index_copy_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                       :: idx
      integer(c_int), value, intent(in)                       :: fid
      logical(c_bool), value, intent(in)                      :: inst
    end subroutine legion_index_copy_launcher_add_dst_field_f

    ! -----------------------------------------------------------------------
    ! Attach Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::AttachLauncher::AttachLauncher()
    function legion_attach_launcher_create_f(logical_region, parent_region, &
        resource) &
        bind(C, name="legion_attach_launcher_create")
      use iso_c_binding
      import legion_attach_launcher_f_t
      import legion_logical_region_f_t
      implicit none

      type(legion_attach_launcher_f_t)                   :: legion_attach_launcher_create_f
      type(legion_logical_region_f_t), value, intent(in) :: logical_region
      type(legion_logical_region_f_t), value, intent(in) :: parent_region
      integer(c_int), value, intent(in) :: resource
    end function legion_attach_launcher_create_f

    ! @see Legion::AttachLauncher::~AttachLauncher()
    subroutine legion_attach_launcher_destroy_f(handle) &
        bind(C, name="legion_attach_launcher_destroy")
      use iso_c_binding
      import legion_attach_launcher_f_t
      implicit none

      type(legion_attach_launcher_f_t), value, intent(in) :: handle
    end subroutine legion_attach_launcher_destroy_f

    ! @see Legion::Runtime::attach_external_resource()
    function legion_attach_launcher_execute_f(runtime, ctx, launcher) &
        bind(C, name="legion_attach_launcher_execute")
      use iso_c_binding
      import legion_physical_region_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_attach_launcher_f_t
      implicit none

      type(legion_physical_region_f_t)                    :: legion_attach_launcher_execute_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_attach_launcher_f_t), value, intent(in) :: launcher
    end function legion_attach_launcher_execute_f

    ! @see Legion::AttachLauncher::attach_array_soa()
    subroutine legion_attach_launcher_add_cpu_soa_field_f(launcher, &
        fid, base_ptr, column_major) &
        bind(C, name="legion_attach_launcher_add_cpu_soa_field")
      use iso_c_binding
      import legion_attach_launcher_f_t
      implicit none

      type(legion_attach_launcher_f_t), value, intent(in) :: launcher
      integer(c_int), value, intent(in)                   :: fid
      type(c_ptr), value, intent(in)                      :: base_ptr
      logical(c_bool), value, intent(in)                  :: column_major
    end subroutine legion_attach_launcher_add_cpu_soa_field_f

    ! @see Legion::Runtime::detach_external_resource()
    function legion_detach_external_resource_f(runtime, ctx, handle) &
        bind(C, name="legion_detach_external_resource")
      use iso_c_binding
      import legion_future_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_future_f_t)                             :: legion_detach_external_resource_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_physical_region_f_t), value, intent(in) :: handle
    end function legion_detach_external_resource_f

    ! @see Legion::Runtime::detach_external_resource()
    function legion_flush_detach_external_resource_f(runtime, ctx, handle, &
        flush) &
        bind(C, name="legion_flush_detach_external_resource")
      use iso_c_binding
      import legion_future_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_future_f_t)                             :: legion_flush_detach_external_resource_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_physical_region_f_t), value, intent(in) :: handle
      logical(c_bool), value, intent(in)                  :: flush
    end function legion_flush_detach_external_resource_f

    ! @see Legion::Runtime::detach_external_resource()
    function legion_unordered_detach_external_resource_f(runtime, ctx, handle, &
        flush, unordered) &
        bind(C, name="legion_unordered_detach_external_resource")
      use iso_c_binding
      import legion_future_f_t
      import legion_runtime_f_t
      import legion_context_f_t
      import legion_physical_region_f_t
      implicit none

      type(legion_future_f_t)                             :: legion_unordered_detach_external_resource_f
      type(legion_runtime_f_t), value, intent(in)         :: runtime
      type(legion_context_f_t), value, intent(in)         :: ctx
      type(legion_physical_region_f_t), value, intent(in) :: handle
      logical(c_bool), value, intent(in)                  :: flush
      logical(c_bool), value, intent(in)                  :: unordered
    end function legion_unordered_detach_external_resource_f

    ! -----------------------------------------------------------------------
    ! Miscellaneous Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::get_runtime()
    function legion_runtime_get_runtime_f() &
       bind(C, name="legion_runtime_get_runtime")
      use iso_c_binding
      import legion_runtime_f_t
      implicit none

      type(legion_runtime_f_t) :: legion_runtime_get_runtime_f
    end function legion_runtime_get_runtime_f

    ! -----------------------------------------------------------------------
    ! Execution Constraints
    ! -----------------------------------------------------------------------
    ! Legion::ExecutionConstraintSet::ExecutionConstraintSet()
    function legion_execution_constraint_set_create_f() &
        bind(C, name="legion_execution_constraint_set_create")
      use iso_c_binding
      import legion_execution_constraint_set_f_t
      implicit none

      type(legion_execution_constraint_set_f_t) :: legion_execution_constraint_set_create_f
    end function legion_execution_constraint_set_create_f

    ! Legion::ExecutionConstraintSet::~ExecutionConstraintSet()
    subroutine legion_execution_constraint_set_destroy_f(handle) &
        bind(C, name="legion_execution_constraint_set_destroy")
      use iso_c_binding
      import legion_execution_constraint_set_f_t
      implicit none

      type(legion_execution_constraint_set_f_t), value, intent(in) :: handle
    end subroutine legion_execution_constraint_set_destroy_f

    ! Legion::ExecutionConstraintSet::add_constraint(Legion::ProcessorConstraint)
    subroutine legion_execution_constraint_set_add_processor_constraint_f(handle, proc_kind) &
        bind(C, name="legion_execution_constraint_set_add_processor_constraint")
      use iso_c_binding
      import legion_execution_constraint_set_f_t
      implicit none

      type(legion_execution_constraint_set_f_t), value, intent(in)    :: handle
      integer(c_int), value, intent(in)                               :: proc_kind
    end subroutine legion_execution_constraint_set_add_processor_constraint_f

    ! -----------------------------------------------------------------------
    ! Task Layout Constraints
    ! -----------------------------------------------------------------------
    ! Legion::TaskLayoutConstraintSet::TaskLayoutConstraintSet()
    function legion_task_layout_constraint_set_create_f() &
        bind(C, name="legion_task_layout_constraint_set_create")
      use iso_c_binding
      import legion_task_layout_constraint_set_f_t
      implicit none

      type(legion_task_layout_constraint_set_f_t) :: legion_task_layout_constraint_set_create_f
    end function legion_task_layout_constraint_set_create_f

    ! Legion::TaskLayoutConstraintSet::TaskLayoutConstraintSet()
    subroutine legion_task_layout_constraint_set_destroy_f(handle) &
        bind(C, name="legion_task_layout_constraint_set_destroy")
      use iso_c_binding
      import legion_task_layout_constraint_set_f_t
      implicit none

      type(legion_task_layout_constraint_set_f_t), value, intent(in) :: handle
    end subroutine legion_task_layout_constraint_set_destroy_f

    ! -----------------------------------------------------------------------
    ! Start-up Operations
    ! -----------------------------------------------------------------------
    ! Legion::Runtime::set_top_level_task_id()
    subroutine legion_runtime_set_top_level_task_id_f(top_id) &
        bind(C, name="legion_runtime_set_top_level_task_id")
      use iso_c_binding
      implicit none

      integer(c_int), value, intent(in) :: top_id
    end subroutine legion_runtime_set_top_level_task_id_f

    ! Legion::Runtime::preregister_task_variant()
    function legion_runtime_preregister_task_variant_fnptr_f(id, variant_id, task_name, &
                                                             variant_name, &
                                                             execution_constraints, &
                                                             layout_constraints, &
                                                             options, &
                                                             wrapped_task_pointer, &
                                                             userdata, &
                                                             userlen) &
        bind(C, name="legion_runtime_preregister_task_variant_fnptr")
      use iso_c_binding
      import legion_execution_constraint_set_f_t
      import legion_task_layout_constraint_set_f_t
      import legion_task_config_options_f_t
      implicit none

      integer(c_int)                                                  :: legion_runtime_preregister_task_variant_fnptr_f
      integer(c_int), value, intent(in)                               :: id
      integer(c_int), value, intent(in)                               :: variant_id
      character(kind=c_char), intent(in)                              :: task_name(*) ! pass reference
      character(kind=c_char), intent(in)                              :: variant_name(*) ! pass reference
      type(legion_execution_constraint_set_f_t), value, intent(in)    :: execution_constraints
      type(legion_task_layout_constraint_set_f_t), value, intent(in)  :: layout_constraints
      type(legion_task_config_options_f_t), value, intent(in)         :: options
      type(c_funptr), value, intent(in)                               :: wrapped_task_pointer
      type(c_ptr), value, intent(in)                                  :: userdata
      integer(c_size_t), value, intent(in)                            :: userlen
    end function legion_runtime_preregister_task_variant_fnptr_f

    ! Legion::Runtime::start()
    function legion_runtime_start_f(argc, argv, background) &
        bind(C, name="legion_runtime_start")
      use iso_c_binding
      implicit none

      integer(c_int)                      :: legion_runtime_start_f
      integer(c_int), value, intent(in)   :: argc
      type(c_ptr), intent(in)             :: argv(*) ! pass reference
      logical(c_bool), value, intent(in)  :: background
    end function legion_runtime_start_f

    ! Legion::LegionTaskWrapper::legion_task_preamble()
    subroutine legion_task_preamble_f(tdata, tdatalen, proc_id, &
                                      task, regionptr, num_regions, &
                                      ctx, runtime) &
        bind(C, name="legion_task_preamble")
      use iso_c_binding
      import legion_task_f_t
      import legion_physical_region_f_t
      import legion_context_f_t
      import legion_runtime_f_t
      implicit none

      type(c_ptr), value, intent(in)                  :: tdata
      integer(c_size_t), value, intent(in)            :: tdatalen
      integer(c_long_long), value, intent(in)         :: proc_id
      type(legion_task_f_t), intent(out)              :: task ! pass reference
      type(c_ptr), intent(out)                        :: regionptr
      integer(c_int), intent(out)                     :: num_regions ! pass reference
      type(legion_context_f_t), intent(out)           :: ctx ! pass reference
      type(legion_runtime_f_t), intent(out)           :: runtime ! pass reference
    end subroutine legion_task_preamble_f

    ! Legion::LegionTaskWrapper::legion_task_postamble()
    subroutine legion_task_postamble_f(runtime, ctx, retval, retsize) &
        bind(C, name="legion_task_postamble")
      use iso_c_binding
      import legion_runtime_f_t
      import legion_context_f_t
      implicit none

      type(legion_runtime_f_t), value, intent(in) :: runtime
      type(legion_context_f_t), value, intent(in) :: ctx
      type(c_ptr), value, intent(in)              :: retval
      integer(c_size_t), value, intent(in)        :: retsize
    end subroutine legion_task_postamble_f

  end interface
end module
