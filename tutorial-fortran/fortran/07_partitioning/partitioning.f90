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
    type(legion_accessor_array_1d_f_t) :: accessor
        
    type(legion_domain_f_t) :: index_domain
    type(legion_index_space_f_t) :: index_space
    type(legion_rect_1d_f_t) :: index_rect, subrect
    type(legion_byte_offset_f_t) :: offset
    real(c_double), target :: x_value
    type(c_ptr) :: x_ptr
    type(legion_point_1d_f_t) :: point
    integer :: i

    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)
                                
    pr = legion_get_physical_region_by_id_f(regionptr, 0, num_regions)
    task_arg_ptr = legion_task_get_args_f(task)
    call c_f_pointer(task_arg_ptr, task_arg)
    arglen = legion_task_get_arglen_f(task)
    fid = task_arg
    
!    if (fid == 0) then
 !       call sleep(5)
  !  end if
    
    accessor = legion_physical_region_get_field_accessor_array_1d_f(pr, fid)
    index_space = legion_task_get_index_space_from_logical_region_f(task, 0)
    index_domain = legion_index_space_get_domain_f(runtime, index_space)
    index_rect = legion_domain_get_rect_1d_f(index_domain)
    
    Print *, "Init Task!", fid, index_rect%lo%x(0), arglen
    do i = index_rect%lo%x(0), index_rect%hi%x(0)
        point%x(0) = i
        x_value = 1.1 * (fid+1) + i
        x_ptr = c_loc(x_value)
        call legion_accessor_array_1d_write_point_f(accessor, point, x_ptr, c_sizeof(x_value))
    end do
    
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
    type(legion_rect_1d_f_t) :: index_rect, subrect
    type(legion_byte_offset_f_t) :: offset
    real(c_double), target :: xy_value, x_value, y_value
    type(c_ptr) :: xy_ptr, x_ptr, y_ptr
    type(legion_point_1d_f_t) :: point
    integer :: i
        
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    pr1 = legion_get_physical_region_by_id_f(regionptr, 0, num_regions)
    pr2 = legion_get_physical_region_by_id_f(regionptr, 1, num_regions)
    task_arg_ptr = legion_task_get_args_f(task)
    call c_f_pointer(task_arg_ptr, task_arg)
    arglen = legion_task_get_arglen_f(task)
    Print *, "Daxpy Task!", task_arg, arglen
    
    accessor_x = legion_physical_region_get_field_accessor_array_1d_f(pr1, 0)
    accessor_y = legion_physical_region_get_field_accessor_array_1d_f(pr1, 1)
    accessor_z = legion_physical_region_get_field_accessor_array_1d_f(pr2, 2)
    index_space = legion_task_get_index_space_from_logical_region_f(task, 0)
    index_domain = legion_index_space_get_domain_f(runtime, index_space)
    index_rect = legion_domain_get_rect_1d_f(index_domain)
    
    do i = index_rect%lo%x(0), index_rect%hi%x(0)
        point%x(0) = i
        x_ptr = c_loc(x_value)
        y_ptr = c_loc(y_value)
        call legion_accessor_array_1d_read_point_f(accessor_x, point, x_ptr, c_sizeof(x_value))
        call legion_accessor_array_1d_read_point_f(accessor_y, point, y_ptr, c_sizeof(y_value))
        xy_value = x_value + y_value
        xy_ptr = c_loc(xy_value)
        call legion_accessor_array_1d_write_point_f(accessor_z, point, xy_ptr, c_sizeof(xy_value))
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
    type(legion_rect_1d_f_t) :: index_rect, subrect
    type(legion_byte_offset_f_t) :: offset
    type(c_ptr) :: raw_ptr_x, raw_ptr_y, raw_ptr_z
    real(c_double), pointer :: x(:), y(:), z(:)
    type(legion_point_1d_f_t) :: point
    type(c_ptr) :: x_ptr, y_ptr, z_ptr
    real(c_double), target :: x_value = 0
    real(c_double), target :: y_value = 0
    real(c_double), target :: z_value = 0
    integer :: i
    logical :: all_passed = .true.
        
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    pr1 = legion_get_physical_region_by_id_f(regionptr, 0, num_regions)
    pr2 = legion_get_physical_region_by_id_f(regionptr, 1, num_regions)
    task_arg_ptr = legion_task_get_args_f(task)
    call c_f_pointer(task_arg_ptr, task_arg)
    arglen = legion_task_get_arglen_f(task)
    Print *, "Check Task!", task_arg, arglen
    
    accessor_x = legion_physical_region_get_field_accessor_array_1d_f(pr1, 0)
    accessor_y = legion_physical_region_get_field_accessor_array_1d_f(pr1, 1)
    accessor_z = legion_physical_region_get_field_accessor_array_1d_f(pr2, 2)
    index_space = legion_task_get_index_space_from_logical_region_f(task, 0)
    index_domain = legion_index_space_get_domain_f(runtime, index_space)
    index_rect = legion_domain_get_rect_1d_f(index_domain)
    
    do i = index_rect%lo%x(0), index_rect%hi%x(0)
        point%x(0) = i
        x_ptr = c_loc(x_value)
        y_ptr = c_loc(y_value)
        z_ptr = c_loc(z_value)
        call legion_accessor_array_1d_read_point_f(accessor_x, point, x_ptr, c_sizeof(x_value))
        call legion_accessor_array_1d_read_point_f(accessor_y, point, y_ptr, c_sizeof(y_value))
        call legion_accessor_array_1d_read_point_f(accessor_z, point, z_ptr, c_sizeof(z_value))
        if (x_value + y_value == z_value) then
        else
            print *, "wrong", i, x_value, y_value, z_value
            all_passed = .false.
        end if
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
    
    type(legion_point_1d_f_t) :: lo, hi, lo_c, hi_c
    type(legion_domain_f_t) :: index_domain, color_domain
    type(legion_rect_1d_f_t) :: index_rect, color_rect
    type(legion_index_space_f_t) :: is, color_is
    type(legion_index_partition_f_t) :: ip
    type(legion_field_space_f_t) :: input_fs, output_fs
    type(legion_logical_region_f_t) :: input_lr, output_lr
    type(legion_field_allocator_f_t) :: ifs_allocator, ofs_allocator
    real(c_double) :: real_number = 0.0
    integer(c_int) :: fid_x, fid_y, fid_z
    integer(c_size_t) :: granularity = 1
    character (len=3), target :: ip_name = "ip"//c_null_char
    character (len=9), target :: input_ip_name = "input_ip"//c_null_char
    character (len=10), target :: output_ip_name = "output_ip"//c_null_char
    type(legion_logical_partition_f_t) :: input_lp, output_lp
    logical(c_bool) :: is_mutable = .false.
    
    type(legion_predicate_f_t) :: pred
    type(legion_task_argument_f_t) :: task_args
    integer(c_int) :: rr_ix, rr_iy, rr_cxy, rr_cz
    logical(c_bool) :: verified = .FALSE.
    logical(c_bool) :: inst = .TRUE.
    type(legion_index_launcher_f_t) :: init_launcher_x, init_launcher_y, daxpy_launcher
    type(legion_task_launcher_f_t) :: check_launcher
    integer(c_long) :: tag = 0
    logical(c_bool) :: must = .FALSE.
    type(legion_future_f_t) :: check_task_future
    type(legion_future_map_f_t) :: init_task_future_map, daxpy_task_future_map
    type(legion_argument_map_f_t) :: arg_map
    
    integer(c_int) :: INIT_TASK_ID=1
    integer(c_int) :: DAXPY_TASK_ID=2
    integer(c_int) :: CHECK_TASK_ID=3
    integer*4, target :: i
    integer*4 :: num_elements = 1024
    integer*4 :: num_subregions = 8
   ! common HELLO_WORLD_TASK_ID
    
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
    index_domain = legion_domain_from_rect_1d_f(index_rect)
    is = legion_index_space_create_domain_f(runtime, ctx, index_domain)
    input_fs = legion_field_space_create_f(runtime, ctx)
    ifs_allocator = legion_field_allocator_create_f(runtime, ctx, input_fs)
    fid_x = legion_field_allocator_allocate_field_f(ifs_allocator, c_sizeof(real_number), 0)
    fid_y = legion_field_allocator_allocate_field_f(ifs_allocator, c_sizeof(real_number), 1)
    call legion_field_allocator_destroy_f(ifs_allocator)
    
    output_fs = legion_field_space_create_f(runtime, ctx)
    ofs_allocator = legion_field_allocator_create_f(runtime, ctx, output_fs)
    fid_z = legion_field_allocator_allocate_field_f(ofs_allocator, c_sizeof(real_number), 2)
    call legion_field_allocator_destroy_f(ofs_allocator)
    print *, fid_x, fid_y, fid_z
    
    input_lr = legion_logical_region_create_f(runtime, ctx, is, input_fs)
    output_lr = legion_logical_region_create_f(runtime, ctx, is, output_fs)
    
    ! create partition
    lo_c%x(0) = 0
    hi_c%x(0) = num_subregions-1
    color_rect%lo = lo_c
    color_rect%hi = hi_c
    color_domain = legion_domain_from_rect_1d_f(color_rect)
    color_is = legion_index_space_create_domain_f(runtime, ctx, color_domain)
    ip = legion_index_partition_create_equal_f(runtime, ctx, is, color_is, granularity, 0)
    call legion_index_partition_attach_name_f(runtime, ip, c_loc(ip_name), is_mutable)
    
    input_lp = legion_logical_partition_create_f(runtime, ctx, input_lr, ip)
    call legion_logical_partition_attach_name_f(runtime, input_lp, c_loc(input_ip_name), is_mutable)
    output_lp = legion_logical_partition_create_f(runtime, ctx, output_lr, ip)
    call legion_logical_partition_attach_name_f(runtime, output_lp, c_loc(output_ip_name), is_mutable)
    
    pred = legion_predicate_true_f() 
    arg_map = legion_argument_map_create_f()
    
    !init task for X
    i = 0
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    init_launcher_x = legion_index_launcher_create_f(INIT_TASK_ID, color_domain, task_args, arg_map, pred, must, 0, tag)
    rr_ix = legion_index_launcher_add_region_requirement_lp_f(init_launcher_x, input_lp, 0, & 
                                                                          WRITE_DISCARD, EXCLUSIVE, &
                                                                          input_lr, tag, verified)
    call legion_index_launcher_add_field_f(init_launcher_x, rr_ix, 0, inst)
    init_task_future_map = legion_index_launcher_execute_f(runtime, ctx, init_launcher_x)
    
    !init task for Y
    i = 1
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    init_launcher_y = legion_index_launcher_create_f(INIT_TASK_ID, color_domain, task_args, arg_map, pred, must, 0, tag)
    rr_iy = legion_index_launcher_add_region_requirement_lp_f(init_launcher_y, input_lp, 0, & 
                                                                          WRITE_DISCARD, EXCLUSIVE, &
                                                                          input_lr, tag, verified)
    call legion_index_launcher_add_field_f(init_launcher_y, rr_iy, 1, inst)
    init_task_future_map = legion_index_launcher_execute_f(runtime, ctx, init_launcher_y)
    
    print *, rr_ix, rr_iy
    
    !daxpy task
    i = 2
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    daxpy_launcher = legion_index_launcher_create_f(DAXPY_TASK_ID, color_domain, task_args, arg_map, pred, must, 0, tag)
    rr_cxy = legion_index_launcher_add_region_requirement_lp_f(daxpy_launcher, input_lp, 0, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       input_lr, tag, verified)
    call legion_index_launcher_add_field_f(daxpy_launcher, rr_cxy, 0, inst)
    call legion_index_launcher_add_field_f(daxpy_launcher, rr_cxy, 1, inst)
    rr_cz = legion_index_launcher_add_region_requirement_lp_f(daxpy_launcher, output_lp, 0, & 
                                                                       WRITE_DISCARD, EXCLUSIVE, &
                                                                       output_lr, tag, verified)
    call legion_index_launcher_add_field_f(daxpy_launcher, rr_cz, 2, inst)
    daxpy_task_future_map = legion_index_launcher_execute_f(runtime, ctx, daxpy_launcher)
    
    !check task
    i = 3
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    check_launcher = legion_task_launcher_create_f(CHECK_TASK_ID, task_args, pred, 0, tag)
    rr_cxy = legion_task_launcher_add_region_requirement_logical_region_f(check_launcher, input_lr, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       input_lr, tag, verified)
    call legion_task_launcher_add_field_f(check_launcher, rr_cxy, 0, inst)
    call legion_task_launcher_add_field_f(check_launcher, rr_cxy, 1, inst)
    rr_cz = legion_task_launcher_add_region_requirement_logical_region_f(check_launcher, output_lr, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       output_lr, tag, verified)
    call legion_task_launcher_add_field_f(check_launcher, rr_cz, 2, inst)
    check_task_future = legion_task_launcher_execute_f(runtime, ctx, check_launcher)
    
    print *, rr_cxy, rr_cz
    
    
    ! clean up
    call legion_logical_region_destroy_f(runtime, ctx, input_lr)
    call legion_logical_region_destroy_f(runtime, ctx, output_lr)
    call legion_field_space_destroy_f(runtime, ctx, input_fs)
    call legion_field_space_destroy_f(runtime, ctx, output_fs)
    call legion_index_space_destroy_f(runtime, ctx, is)
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