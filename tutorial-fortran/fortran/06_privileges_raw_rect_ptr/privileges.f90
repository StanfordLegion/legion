module daxpy
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID = 0
  integer(c_int), parameter :: INIT_TASK_ID=1
  integer(c_int), parameter :: DAXPY_TASK_ID=2
  integer(c_int), parameter :: CHECK_TASK_ID=3
  
contains
  subroutine init_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(legion_task_f_t) :: task
    integer(c_int) :: num_regions
    type(legion_context_f_t) :: ctx
    type(legion_runtime_f_t) :: runtime
    type(c_ptr) :: regionptr
    integer(c_size_t) :: retsize = 0
    integer(c_size_t) :: arglen
    integer*4, pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    type(legion_physical_region_f_t) :: pr
    integer(c_int) :: fid
    type(legion_accessor_array_1d_f_t) :: accessor
        
    type(legion_domain_f_t) :: index_domain
    type(legion_index_space_f_t) :: index_space
    integer(c_size_t) :: index_size
    type(legion_rect_1d_f_t) :: index_rect, subrect
    type(legion_byte_offset_f_t) :: offset
    type(c_ptr) :: raw_ptr
    real(kind=8), pointer :: x(:)
    integer :: i

    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)
                                
    call legion_get_physical_region_by_id_f(regionptr, 0, num_regions, pr)
    call legion_task_get_args_f(task, task_arg_ptr)
    call c_f_pointer(task_arg_ptr, task_arg)
    call legion_task_get_arglen_f(task, arglen)
    fid = task_arg
    Print *, "Init Task!", fid, arglen
    
    call legion_physical_region_get_field_accessor_array_1d_f(pr, fid, accessor)
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_1d_f(index_domain, index_rect)
    call legion_domain_get_volume_f(index_domain, index_size)
    
    call legion_accessor_array_1d_raw_rect_ptr_f(accessor, index_rect, subrect, offset, raw_ptr)
    call c_f_pointer(raw_ptr, x, [index_size-1])
    print *, raw_ptr
    
    ! fortran array starts from 1
    do i = 1, index_size
      x(i) = 1.1 * (fid+1)
    end do
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
  end subroutine init_task

  subroutine daxpy_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    type(legion_accessor_array_1d_f_t) :: accessor_x, accessor_y, accessor_z
    
    type(legion_task_f_t) :: task
    integer(c_int) :: num_regions
    type(legion_context_f_t) :: ctx
    type(legion_runtime_f_t) :: runtime
    type(c_ptr) :: regionptr
    integer(c_size_t) :: retsize = 0
    integer(c_size_t) :: arglen
    integer*4, pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    
    type(legion_physical_region_f_t) :: pr1, pr2
        
    type(legion_domain_f_t) :: index_domain
    type(legion_index_space_f_t) :: index_space
    integer(c_size_t) :: index_size
    type(legion_rect_1d_f_t) :: index_rect, subrect
    type(legion_byte_offset_f_t) :: offset
    type(c_ptr) :: raw_ptr_x, raw_ptr_y, raw_ptr_z
    real(kind=8), pointer :: x(:), y(:), z(:)
    integer :: i
        
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    call legion_get_physical_region_by_id_f(regionptr, 0, num_regions, pr1)
    call legion_get_physical_region_by_id_f(regionptr, 1, num_regions, pr2)
    call legion_task_get_args_f(task, task_arg_ptr)
    call c_f_pointer(task_arg_ptr, task_arg)
    call legion_task_get_arglen_f(task, arglen)
    Print *, "Daxpy Task!", task_arg, arglen
    
    call legion_physical_region_get_field_accessor_array_1d_f(pr1, 0, accessor_x)
    call legion_physical_region_get_field_accessor_array_1d_f(pr1, 1, accessor_y)
    call legion_physical_region_get_field_accessor_array_1d_f(pr2, 2, accessor_z)
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_1d_f(index_domain, index_rect)
    call legion_domain_get_volume_f(index_domain, index_size)
    
    call legion_accessor_array_1d_raw_rect_ptr_f(accessor_x, index_rect, subrect, offset, raw_ptr_x)
    call legion_accessor_array_1d_raw_rect_ptr_f(accessor_y, index_rect, subrect, offset, raw_ptr_y)
    call legion_accessor_array_1d_raw_rect_ptr_f(accessor_z, index_rect, subrect, offset, raw_ptr_z)
    print *, raw_ptr_x, raw_ptr_y, raw_ptr_z
    
    call c_f_pointer(raw_ptr_x, x, [index_size-1])
    call c_f_pointer(raw_ptr_y, y, [index_size-1])
    call c_f_pointer(raw_ptr_z, z, [index_size-1])
    
    do i = 1, index_size
      z(i) = x(i) + y(i)
    end do
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
  end subroutine daxpy_task

  subroutine check_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    type(legion_accessor_array_1d_f_t) :: accessor_x, accessor_y, accessor_z
    
    type(legion_task_f_t) :: task
    integer(c_int) :: num_regions
    type(legion_context_f_t) :: ctx
    type(legion_runtime_f_t) :: runtime
    type(c_ptr) :: regionptr
    integer(c_size_t) :: retsize = 0
    integer(c_size_t) :: arglen
    integer*4, pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    
    type(legion_physical_region_f_t) :: pr1, pr2
        
    type(legion_domain_f_t) :: index_domain
    type(legion_index_space_f_t) :: index_space
    integer(c_size_t) :: index_size
    type(legion_rect_1d_f_t) :: index_rect, subrect
    type(legion_byte_offset_f_t) :: offset
    type(c_ptr) :: raw_ptr_x, raw_ptr_y, raw_ptr_z
    real(kind=8), pointer :: x(:), y(:), z(:)
    integer :: i
    logical :: all_passed = .true.
        
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    call legion_get_physical_region_by_id_f(regionptr, 0, num_regions, pr1)
    call legion_get_physical_region_by_id_f(regionptr, 1, num_regions, pr2)
    call legion_task_get_args_f(task, task_arg_ptr)
    call c_f_pointer(task_arg_ptr, task_arg)
    call legion_task_get_arglen_f(task, arglen)
    Print *, "Check Task!", task_arg, arglen
    
    call legion_physical_region_get_field_accessor_array_1d_f(pr1, 0, accessor_x)
    call legion_physical_region_get_field_accessor_array_1d_f(pr1, 1, accessor_y)
    call legion_physical_region_get_field_accessor_array_1d_f(pr2, 2, accessor_z)
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_1d_f(index_domain, index_rect)
    call legion_domain_get_volume_f(index_domain, index_size)
    
    call legion_accessor_array_1d_raw_rect_ptr_f(accessor_x, index_rect, subrect, offset, raw_ptr_x)
    call legion_accessor_array_1d_raw_rect_ptr_f(accessor_y, index_rect, subrect, offset, raw_ptr_y)
    call legion_accessor_array_1d_raw_rect_ptr_f(accessor_z, index_rect, subrect, offset, raw_ptr_z)
    print *, raw_ptr_x, raw_ptr_y, raw_ptr_z, index_size
    
    call c_f_pointer(raw_ptr_x, x, [index_size-1])
    call c_f_pointer(raw_ptr_y, y, [index_size-1])
    call c_f_pointer(raw_ptr_z, z, [index_size-1])
    
    do i = 1, index_size
      if (x(i) + y(i) == z(i)) then
      else
        print *, "wrong", x(i), y(i), z(i)
        all_passed = .false.
      end if
    end do
    
    if (all_passed .eqv. .true.) then
      print *, "Pass"
    else
      print *, "Failed"
    end if
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
  end subroutine check_task

  subroutine top_level_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(legion_task_f_t) :: task
    integer(c_int) :: num_regions
    type(legion_context_f_t) :: ctx
    type(legion_runtime_f_t) :: runtime
    type(c_ptr) :: regionptr
    integer(c_size_t) :: retsize = 0
    
    type(legion_point_1d_f_t) :: lo, hi
    type(legion_domain_f_t) :: index_domain
    type(legion_rect_1d_f_t) :: index_rect
    type(legion_index_space_f_t) :: is
    type(legion_field_space_f_t) :: input_fs, output_fs
    type(legion_logical_region_f_t) :: input_lr, output_lr
    type(legion_field_allocator_f_t) :: ifs_allocator, ofs_allocator
    real*8 :: real_number = 0.0
    integer(c_int) :: fid_x, fid_y, fid_z
    
    type(legion_predicate_f_t) :: pred
    type(legion_task_argument_f_t) :: task_args
    integer(c_int) :: rr_ix, rr_iy, rr_cxy, rr_cz
    type(legion_task_launcher_f_t) :: init_launcher_x, init_launcher_y, daxpy_launcher, check_launcher
    integer(c_long) :: tag = 0
    type(legion_future_f_t) :: init_task_future, daxpy_task_future, check_task_future
    
    integer*4, target :: i
    integer*4 :: num_elements = 2048
    
    Print *, "TOP Level Task!"
    
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    ! create index space, field space and logical region
    lo%x(0) = 0
    hi%x(0) = num_elements-1
    index_rect%lo = lo
    index_rect%hi = hi
    call legion_domain_from_rect_1d_f(index_rect, index_domain)
    call legion_index_space_create_domain_f(runtime, ctx, index_domain, is)
    call legion_field_space_create_f(runtime, ctx, input_fs)
    call legion_field_allocator_create_f(runtime, ctx, input_fs, ifs_allocator)
    call legion_field_allocator_allocate_field_f(ifs_allocator, c_sizeof(real_number), 0, fid_x)
    call legion_field_allocator_allocate_field_f(ifs_allocator, c_sizeof(real_number), 1, fid_y)
    call legion_field_allocator_destroy_f(ifs_allocator)
    
    call legion_field_space_create_f(runtime, ctx, output_fs)
    call legion_field_allocator_create_f(runtime, ctx, output_fs, ofs_allocator)
    call legion_field_allocator_allocate_field_f(ofs_allocator, c_sizeof(real_number), 2, fid_z)
    call legion_field_allocator_destroy_f(ofs_allocator)
    print *, fid_x, fid_y, fid_z
    
    call legion_logical_region_create_f(runtime, ctx, is, input_fs, input_lr)
    call legion_logical_region_create_f(runtime, ctx, is, output_fs, output_lr)
    
    call legion_predicate_true_f(pred) 
    
    !init task for X
    i = 0
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_task_launcher_create_f(INIT_TASK_ID, task_args, pred, 0, tag, init_launcher_x)
    call legion_task_launcher_add_region_requirement_logical_region_f(init_launcher_x, input_lr, & 
                                                                       WRITE_DISCARD, EXCLUSIVE, &
                                                                       input_lr, tag, .false., rr_ix)
    call legion_task_launcher_add_field_f(init_launcher_x, rr_ix, fid_x, .true.)
    call legion_task_launcher_execute_f(runtime, ctx, init_launcher_x, init_task_future)
    
    !init task for Y
    i = 1
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_task_launcher_create_f(INIT_TASK_ID, task_args, pred, 0, tag, init_launcher_y)
    call legion_task_launcher_add_region_requirement_logical_region_f(init_launcher_y, input_lr, & 
                                                                       WRITE_DISCARD, EXCLUSIVE, &
                                                                       input_lr, tag, .false., rr_iy)
    call legion_task_launcher_add_field_f(init_launcher_y, rr_iy, fid_y, .true.)
    call legion_task_launcher_execute_f(runtime, ctx, init_launcher_y, init_task_future)
    
    !daxpy task
    i = 2
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_task_launcher_create_f(DAXPY_TASK_ID, task_args, pred, 0, tag, daxpy_launcher)
    call legion_task_launcher_add_region_requirement_logical_region_f(daxpy_launcher, input_lr, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       input_lr, tag, .false., rr_cxy)
    call legion_task_launcher_add_field_f(daxpy_launcher, rr_cxy, fid_x, .true.)
    call legion_task_launcher_add_field_f(daxpy_launcher, rr_cxy, fid_y, .true.)
    call legion_task_launcher_add_region_requirement_logical_region_f(daxpy_launcher, output_lr, & 
                                                                       WRITE_DISCARD, EXCLUSIVE, &
                                                                       output_lr, tag, .false., rr_cz)
    call legion_task_launcher_add_field_f(daxpy_launcher, rr_cz, fid_z, .true.)
    call legion_task_launcher_execute_f(runtime, ctx, daxpy_launcher, daxpy_task_future)
    
    !check task
    i = 3
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_task_launcher_create_f(CHECK_TASK_ID, task_args, pred, 0, tag, check_launcher)
    call legion_task_launcher_add_region_requirement_logical_region_f(check_launcher, input_lr, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       input_lr, tag, .false., rr_cxy)
    call legion_task_launcher_add_field_f(check_launcher, rr_cxy, fid_x, .true.)
    call legion_task_launcher_add_field_f(check_launcher, rr_cxy, fid_y, .true.)
    call legion_task_launcher_add_region_requirement_logical_region_f(check_launcher, output_lr, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       output_lr, tag, .false., rr_cz)
    call legion_task_launcher_add_field_f(check_launcher, rr_cz, fid_z, .true.)
    call legion_task_launcher_execute_f(runtime, ctx, check_launcher, check_task_future)
    
    ! clean up
    call legion_logical_region_destroy_f(runtime, ctx, input_lr)
    call legion_logical_region_destroy_f(runtime, ctx, output_lr)
    call legion_field_space_destroy_f(runtime, ctx, input_fs)
    call legion_field_space_destroy_f(runtime, ctx, output_fs)
    call legion_index_space_destroy_f(runtime, ctx, is)
    call legion_task_launcher_destroy_f(init_launcher_x)
    call legion_task_launcher_destroy_f(init_launcher_y)
    call legion_task_launcher_destroy_f(daxpy_launcher)
    call legion_task_launcher_destroy_f(check_launcher)
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
  end subroutine top_level_task
end module daxpy

Program daxpy_raw_rect_pointer
    use legion_fortran
    use iso_c_binding
    use daxpy
    implicit none
    
    type(legion_execution_constraint_set_f_t) :: execution_constraints
    type(legion_task_layout_constraint_set_f_t) :: layout_constraints
    type(legion_task_config_options_f_t) :: config_options
    integer(c_int) :: task_id_1, task_id_2, task_id_3, task_id_4
    integer(c_size_t) :: userlen = 0
    integer(c_int) :: runtime_start_rv
    type(c_funptr) :: c_func_ptr
        
    Print *, "Hello World from Main!"
    call legion_runtime_set_top_level_task_id_f(TOP_LEVEL_TASK_ID)
    call legion_execution_constraint_set_create_f(execution_constraints)
    call legion_execution_constraint_set_add_processor_constraint_f(execution_constraints, LOC_PROC)
    call legion_task_layout_constraint_set_create_f(layout_constraints)
    config_options%leaf = .false.
    config_options%inner = .false.
    config_options%idempotent = .false.
    
    c_func_ptr = c_funloc(top_level_task)
    
    call legion_runtime_preregister_task_variant_fnptr_f(TOP_LEVEL_TASK_ID, c_char_"top_level_task"//c_null_char, &
                                                        c_char_"cpu_variant"//c_null_char, &
                                                        execution_constraints, &
                                                        layout_constraints, &
                                                        config_options, &
                                                        c_func_ptr, &
                                                        c_null_ptr, &
                                                        userlen, task_id_1)
    
    c_func_ptr = c_funloc(init_task)

    call legion_runtime_preregister_task_variant_fnptr_f(INIT_TASK_ID, c_char_"init_task"//c_null_char, &
                                                        c_char_"cpu_variant"//c_null_char, &
                                                        execution_constraints, &
                                                        layout_constraints, &
                                                        config_options, &
                                                        c_func_ptr, &
                                                        c_null_ptr, &
                                                        userlen, task_id_2)
                                                                
    c_func_ptr = c_funloc(daxpy_task)

    call legion_runtime_preregister_task_variant_fnptr_f(DAXPY_TASK_ID, c_char_"daxpy_task"//c_null_char, &
                                                        c_char_"cpu_variant"//c_null_char, &
                                                        execution_constraints, &
                                                        layout_constraints, &
                                                        config_options, &
                                                        c_func_ptr, &
                                                        c_null_ptr, &
                                                        userlen, task_id_3)
    
    c_func_ptr = c_funloc(check_task)

    call legion_runtime_preregister_task_variant_fnptr_f(CHECK_TASK_ID, c_char_"check_task"//c_null_char, &
                                                        c_char_"cpu_variant"//c_null_char, &
                                                        execution_constraints, &
                                                        layout_constraints, &
                                                        config_options, &
                                                        c_func_ptr, &
                                                        c_null_ptr, &
                                                        userlen, task_id_4)
    call legion_runtime_start_f(0, c_null_ptr, .false., runtime_start_rv)
End Program daxpy_raw_rect_pointer
