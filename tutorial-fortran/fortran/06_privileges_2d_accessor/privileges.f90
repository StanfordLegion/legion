function init_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    integer(c_int) ::init_task
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
    type(legion_accessor_array_2d_f_t) :: accessor
        
    type(legion_domain_f_t) :: index_domain
    type(legion_index_space_f_t) :: index_space
    type(legion_point_2d_f_t) :: lo, hi
    type(legion_rect_2d_f_t) :: index_rect
    real(c_double), target :: x_value
    type(c_ptr) :: x_ptr
    type(legion_point_2d_f_t) :: point
    integer :: i, j

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
    
!    if (fid == 0) then
 !       call sleep(5)
  !  end if
    
    call legion_physical_region_get_field_accessor_array_2d_f(pr, fid, accessor)
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_2d_f(index_domain, index_rect)
    lo = index_rect%lo
    hi = index_rect%hi

    do i = lo%x(0), hi%x(0)
        do j = lo%x(1), hi%x(1)
            point%x(0) = j
            point%x(1) = i
            x_value = 1.1 * (fid+1)
            x_ptr = c_loc(x_value)
            call legion_accessor_array_2d_write_point_f(accessor, point, x_ptr, c_sizeof(x_value))
        end do
    end do
    
    print *, "Init done", hi%x(0)
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
    init_task = 0
end function

function daxpy_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    integer(c_int) ::daxpy_task
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    type(legion_accessor_array_2d_f_t) :: accessor_x, accessor_y, accessor_z
    
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
    type(legion_point_2d_f_t) :: lo, hi
    type(legion_rect_2d_f_t) :: index_rect
    real(c_double), target :: xy_value, x_value, y_value
    type(c_ptr) :: xy_ptr, x_ptr, y_ptr
    type(legion_point_2d_f_t) :: point
    integer :: i, j
        
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
    
    call legion_physical_region_get_field_accessor_array_2d_f(pr1, 0, accessor_x)
    call legion_physical_region_get_field_accessor_array_2d_f(pr1, 1, accessor_y)
    call legion_physical_region_get_field_accessor_array_2d_f(pr2, 2, accessor_z)
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_2d_f(index_domain, index_rect)
    lo = index_rect%lo
    hi = index_rect%hi
    
    do i = lo%x(0), hi%x(0)
        do j = lo%x(1), hi%x(1)
            point%x(0) = j
            point%x(1) = i
            x_ptr = c_loc(x_value)
            y_ptr = c_loc(y_value)
            call legion_accessor_array_2d_read_point_f(accessor_x, point, x_ptr, c_sizeof(x_value))
            call legion_accessor_array_2d_read_point_f(accessor_y, point, y_ptr, c_sizeof(y_value))
            xy_value = x_value + y_value
            xy_ptr = c_loc(xy_value)
            call legion_accessor_array_2d_write_point_f(accessor_z, point, xy_ptr, c_sizeof(xy_value))
        end do
    end do
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
    daxpy_task = 0
end function

function check_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    integer(c_int) ::check_task
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    type(legion_accessor_array_2d_f_t) :: accessor_x, accessor_y, accessor_z
    
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
    type(legion_point_2d_f_t) :: lo, hi
    type(legion_rect_2d_f_t) :: index_rect
    type(legion_point_2d_f_t) :: point
    type(c_ptr) :: x_ptr, y_ptr, z_ptr
    real(c_double), target :: x_value = 0
    real(c_double), target :: y_value = 0
    real(c_double), target :: z_value = 0
    integer :: i, j
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
    
    call legion_physical_region_get_field_accessor_array_2d_f(pr1, 0, accessor_x)
    call legion_physical_region_get_field_accessor_array_2d_f(pr1, 1, accessor_y)
    call legion_physical_region_get_field_accessor_array_2d_f(pr2, 2, accessor_z)
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_2d_f(index_domain, index_rect)
    lo = index_rect%lo
    hi = index_rect%hi
    
    do i = lo%x(0), hi%x(0)
        do j = lo%x(1), hi%x(1)
            point%x(0) = i
            point%x(1) = j
            x_ptr = c_loc(x_value)
            y_ptr = c_loc(y_value)
            z_ptr = c_loc(z_value)
            call legion_accessor_array_2d_read_point_f(accessor_x, point, x_ptr, c_sizeof(x_value))
            call legion_accessor_array_2d_read_point_f(accessor_y, point, y_ptr, c_sizeof(y_value))
            call legion_accessor_array_2d_read_point_f(accessor_z, point, z_ptr, c_sizeof(z_value))
            if (x_value + y_value == z_value) then
            else
                print *, "wrong", i, x_value, y_value, z_value
                all_passed = .false.
            end if
        end do
    end do
    
    if (all_passed .eqv. .true.) then
        print *, "Pass"
    else
        print *, "Failed"
    end if
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
    check_task = 0
end function

function top_level_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    integer(c_int) ::top_level_task
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
    
    type(legion_point_2d_f_t) :: lo, hi
    type(legion_domain_f_t) :: index_domain
    type(legion_rect_2d_f_t) :: index_rect
    type(legion_index_space_f_t) :: is
    type(legion_field_space_f_t) :: input_fs, output_fs
    type(legion_logical_region_f_t) :: input_lr, output_lr
    type(legion_field_allocator_f_t) :: ifs_allocator, ofs_allocator
    real(c_double) :: real_number = 0.0
    integer(c_int) :: fid_x, fid_y, fid_z
    
    type(legion_predicate_f_t) :: pred
    type(legion_task_argument_f_t) :: task_args
    integer(c_int) :: rr_ix, rr_iy, rr_cxy, rr_cz
    logical(c_bool) :: verified = .FALSE.
    logical(c_bool) :: inst = .TRUE.
    type(legion_task_launcher_f_t) :: init_launcher_x, init_launcher_y, daxpy_launcher, check_launcher
    integer(c_long) :: tag = 0
    type(legion_future_f_t) :: init_task_future, daxpy_task_future, check_task_future
    
    integer(c_int) :: INIT_TASK_ID=1
    integer(c_int) :: DAXPY_TASK_ID=2
    integer(c_int) :: CHECK_TASK_ID=3
    integer*4, target :: i
    integer*4 :: num_elements = 1024
   ! common HELLO_WORLD_TASK_ID
    
    Print *, "TOP Level Task!"
    
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    ! create index space, field space and logical region
    lo%x(0) = 0
    lo%x(1) = 0
    hi%x(0) = num_elements-1
    hi%x(1) = num_elements-1
    index_rect%lo = lo
    index_rect%hi = hi
    call legion_domain_from_rect_2d_f(index_rect, index_domain)
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
                                                                       input_lr, tag, verified, rr_ix)
    call legion_task_launcher_add_field_f(init_launcher_x, rr_ix, 0, inst)
    call legion_task_launcher_execute_f(runtime, ctx, init_launcher_x, init_task_future)
    
    !init task for Y
    i = 1
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_task_launcher_create_f(INIT_TASK_ID, task_args, pred, 0, tag, init_launcher_y)
    call legion_task_launcher_add_region_requirement_logical_region_f(init_launcher_y, input_lr, & 
                                                                       WRITE_DISCARD, EXCLUSIVE, &
                                                                       input_lr, tag, verified, rr_iy)
    call legion_task_launcher_add_field_f(init_launcher_y, rr_iy, 1, inst)
    call legion_task_launcher_execute_f(runtime, ctx, init_launcher_y, init_task_future)
    
    print *, rr_ix, rr_iy
    
    !daxpy task
    i = 2
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_task_launcher_create_f(DAXPY_TASK_ID, task_args, pred, 0, tag, daxpy_launcher)
    call legion_task_launcher_add_region_requirement_logical_region_f(daxpy_launcher, input_lr, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       input_lr, tag, verified, rr_cxy)
    call legion_task_launcher_add_field_f(daxpy_launcher, rr_cxy, 0, inst)
    call legion_task_launcher_add_field_f(daxpy_launcher, rr_cxy, 1, inst)
    call legion_task_launcher_add_region_requirement_logical_region_f(daxpy_launcher, output_lr, & 
                                                                       WRITE_DISCARD, EXCLUSIVE, &
                                                                       output_lr, tag, verified, rr_cz)
    call legion_task_launcher_add_field_f(daxpy_launcher, rr_cz, 2, inst)
    call legion_task_launcher_execute_f(runtime, ctx, daxpy_launcher, daxpy_task_future)
    
    !check task
    i = 3
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_task_launcher_create_f(CHECK_TASK_ID, task_args, pred, 0, tag, check_launcher)
    call legion_task_launcher_add_region_requirement_logical_region_f(check_launcher, input_lr, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       input_lr, tag, verified, rr_cxy)
    call legion_task_launcher_add_field_f(check_launcher, rr_cxy, 0, inst)
    call legion_task_launcher_add_field_f(check_launcher, rr_cxy, 1, inst)
    call legion_task_launcher_add_region_requirement_logical_region_f(check_launcher, output_lr, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       output_lr, tag, verified, rr_cz)
    call legion_task_launcher_add_field_f(check_launcher, rr_cz, 2, inst)
    call legion_task_launcher_execute_f(runtime, ctx, check_launcher, check_task_future)
    
    print *, rr_cxy, rr_cz
    
    
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
    top_level_task = 0
end function

Program Hello
    use legion_fortran
    use iso_c_binding
    implicit none
    type(legion_execution_constraint_set_f_t) :: execution_constraints
    type(legion_task_layout_constraint_set_f_t) :: layout_constraints
    type(legion_task_config_options_f_t) :: config_options
    integer(c_int) :: proc_kind = 2
    integer(c_int) :: TOP_LEVEL_TASK_ID, INIT_TASK_ID, DAXPY_TASK_ID, CHECK_TASK_ID
    integer(c_int) :: task_id_1, task_id_2, task_id_3
    integer(c_size_t) :: userlen = 0
    integer(c_int) :: runtime_start_rv
    logical(c_bool) :: background = .FALSE.
    type(c_funptr) :: c_func_ptr
    
    external top_level_task
    external init_task
    external check_task
    external daxpy_task
    
   ! common TOP_LEVEL_TASK_ID
    !common HELLO_WORLD_TASK_ID
    TOP_LEVEL_TASK_ID = 0
    INIT_TASK_ID = 1
    DAXPY_TASK_ID = 2
    CHECK_TASK_ID = 3
        
    Print *, "Hello World from Main!"
    call legion_runtime_set_top_level_task_id_f(TOP_LEVEL_TASK_ID)
    execution_constraints = legion_execution_constraint_set_create_f()
    call legion_execution_constraint_set_add_processor_constraint_f(execution_constraints, proc_kind)
    layout_constraints = legion_task_layout_constraint_set_create_f()
    config_options%leaf = .FALSE.
    config_options%inner = .FALSE.
    config_options%idempotent = .FALSE.
    
    c_func_ptr = c_funloc(top_level_task)
    
    task_id_1 = legion_runtime_preregister_task_variant_fnptr_f(TOP_LEVEL_TASK_ID, c_char_"top_level_task"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
    
    c_func_ptr = c_funloc(init_task)

    task_id_2 = legion_runtime_preregister_task_variant_fnptr_f(INIT_TASK_ID, c_char_"init_task"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
                                                                
    c_func_ptr = c_funloc(daxpy_task)

    task_id_3 = legion_runtime_preregister_task_variant_fnptr_f(DAXPY_TASK_ID, c_char_"daxpy_task"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
    
    c_func_ptr = c_funloc(check_task)

    task_id_3 = legion_runtime_preregister_task_variant_fnptr_f(CHECK_TASK_ID, c_char_"check_task"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
    runtime_start_rv = legion_runtime_start_f(0, c_null_ptr, background)
End Program Hello