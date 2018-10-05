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
    type(legion_rect_1d_f_t) :: index_rect
    real(c_double), target :: x_value
    type(c_ptr) :: x_ptr
    type(legion_point_1d_f_t) :: point
    integer :: i
    
    type(legion_region_requirement_f_t) :: rr
    integer(c_int) :: rrfid

    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)
                                
    call legion_get_physical_region_by_id_f(regionptr, 0, num_regions, pr)
    call legion_task_get_args_f(task, task_arg_ptr)
    call c_f_pointer(task_arg_ptr, task_arg)
    call legion_task_get_arglen_f(task, arglen)
    fid = task_arg
    
!    if (fid == 0) then
 !       call sleep(5)
  !  end if
  
    call legion_task_get_region_f(task, 0, rr)
    call legion_region_requirement_get_privilege_field_f(rr, 0, rrfid)
    
    
    call legion_physical_region_get_field_accessor_array_1d_f(pr, fid, accessor)
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_1d_f(index_domain, index_rect)
    
    Print *, "Init Task!", rrfid, fid, index_rect%lo%x(0), arglen
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
    use legion_fortran_object_oriented
    use iso_c_binding
    implicit none
    
    integer(c_int) ::daxpy_task
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    type(legion_accessor_array_1d_f_t) :: accessor_x
    type(LegionFieldAccessor1D) :: accessor_y, accessor_z
    type(LegionPoint1D) ::point_1d
    
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
    type(legion_rect_1d_f_t) :: index_rect
    real(c_double), target :: xy_value, x_value, y_value
    type(c_ptr) :: xy_ptr, x_ptr, y_ptr
    type(legion_point_1d_f_t) :: point
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
   ! call legion_physical_region_get_field_accessor_array_1d_f(pr1, 1, accessor_y)
    accessor_y = LegionFieldAccessor1D(pr1, 1, c_sizeof(y_value))
    accessor_z = LegionFieldAccessor1D(pr2, 2, c_sizeof(xy_value))
    
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_1d_f(index_domain, index_rect)
    
    do i = index_rect%lo%x(0), index_rect%hi%x(0)
        point%x(0) = i
        point_1d = LegionPoint1D(i)
        x_ptr = c_loc(x_value)
        y_ptr = c_loc(y_value)
        call legion_accessor_array_1d_read_point_f(accessor_x, point, x_value)
        !call legion_accessor_array_1d_read_point_f(accessor_y, point, y_ptr, c_sizeof(y_value))
        call accessor_y%read_point(point_1d, y_value)
        xy_value = x_value + y_value
        xy_ptr = c_loc(xy_value)
        call accessor_z%write_point(point_1d, xy_value)
        !call legion_accessor_array_1d_write_point_f(accessor_z, point, xy_ptr, c_sizeof(xy_value))
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
    type(legion_rect_1d_f_t) :: index_rect
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
    type(legion_field_space_f_t) :: input_fs, output_fs, cp_fs
    type(legion_logical_region_f_t) :: input_lr, output_lr, cp_lr
    type(legion_field_allocator_f_t) :: ifs_allocator, ofs_allocator, cpfs_allocator
    real(c_double) :: real_number = 0.0
    integer(c_int) :: fid_x, fid_y, fid_z, fid_cpz
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
        
    character (len=10) :: hdf5_file_name = "hdf5_file"//c_null_char
    character (len=13) :: hdf5_dataset_name = "hdf5_dataset"//c_null_char
    logical(c_bool) :: hdf5_file_is_valid
    type(legion_field_map_f_t) :: hdf5_field_map
    type(legion_physical_region_f_t) :: cp_pr
    type(legion_copy_launcher_f_t) :: cp_launcher
    integer(c_int) :: rridx_cp
    logical(c_bool), external :: generate_hdf_file
    
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
    
    call legion_logical_region_create_f(runtime, ctx, is, input_fs, input_lr)
    call legion_logical_region_create_f(runtime, ctx, is, output_fs, output_lr)
    
    ! create logical region for hdf5 file
    call legion_field_space_create_f(runtime, ctx, cp_fs)
    call legion_field_allocator_create_f(runtime, ctx, cp_fs, cpfs_allocator)
    call legion_field_allocator_allocate_field_f(cpfs_allocator, c_sizeof(real_number), 3, fid_cpz)
    call legion_field_allocator_destroy_f(cpfs_allocator)
    call legion_logical_region_create_f(runtime, ctx, is, cp_fs, cp_lr)
    
    print *, fid_x, fid_y, fid_z, fid_cpz
    
    ! create partition
    lo_c%x(0) = 0
    hi_c%x(0) = num_subregions-1
    color_rect%lo = lo_c
    color_rect%hi = hi_c
    call legion_domain_from_rect_1d_f(color_rect, color_domain)
    call legion_index_space_create_domain_f(runtime, ctx, color_domain, color_is)
    call legion_index_partition_create_equal_f(runtime, ctx, is, color_is, granularity, 0, ip)
    call legion_index_partition_attach_name_f(runtime, ip, ip_name, is_mutable)
    
    call legion_logical_partition_create_f(runtime, ctx, input_lr, ip, input_lp)
    call legion_logical_partition_attach_name_f(runtime, input_lp, input_ip_name, is_mutable)
    call legion_logical_partition_create_f(runtime, ctx, output_lr, ip, output_lp)
    call legion_logical_partition_attach_name_f(runtime, output_lp, output_ip_name, is_mutable)
    
    call legion_predicate_true_f(pred) 
    call legion_argument_map_create_f(arg_map)
    
    !init task for X
    i = 0
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_index_launcher_create_f(INIT_TASK_ID, color_domain, task_args, arg_map, pred, must, 0, tag, init_launcher_x)
    call legion_index_launcher_add_region_requirement_lp_f(init_launcher_x, input_lp, 0, & 
                                                                          WRITE_DISCARD, EXCLUSIVE, &
                                                                          input_lr, tag, verified, rr_ix)
    call legion_index_launcher_add_field_f(init_launcher_x, rr_ix, 0, inst)
    call legion_index_launcher_execute_f(runtime, ctx, init_launcher_x, init_task_future_map)
    
    !init task for Y
    i = 1
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_index_launcher_create_f(INIT_TASK_ID, color_domain, task_args, arg_map, pred, must, 0, tag, init_launcher_y)
    call legion_index_launcher_add_region_requirement_lp_f(init_launcher_y, input_lp, 0, & 
                                                                          WRITE_DISCARD, EXCLUSIVE, &
                                                                          input_lr, tag, verified, rr_iy)
    call legion_index_launcher_add_field_f(init_launcher_y, rr_iy, 1, inst)
    call legion_index_launcher_execute_f(runtime, ctx, init_launcher_y, init_task_future_map)
    
    print *, rr_ix, rr_iy
    
    !daxpy task
    i = 2
    task_args%args = c_loc(i)
    task_args%arglen = c_sizeof(i)
    call legion_index_launcher_create_f(DAXPY_TASK_ID, color_domain, task_args, arg_map, pred, must, 0, tag, daxpy_launcher)
    call legion_index_launcher_add_region_requirement_lp_f(daxpy_launcher, input_lp, 0, & 
                                                                       READ_ONLY, EXCLUSIVE, &
                                                                       input_lr, tag, verified, rr_cxy)
    call legion_index_launcher_add_field_f(daxpy_launcher, rr_cxy, 0, inst)
    call legion_index_launcher_add_field_f(daxpy_launcher, rr_cxy, 1, inst)
    call legion_index_launcher_add_region_requirement_lp_f(daxpy_launcher, output_lp, 0, & 
                                                                       WRITE_DISCARD, EXCLUSIVE, &
                                                                       output_lr, tag, verified, rr_cz)
    call legion_index_launcher_add_field_f(daxpy_launcher, rr_cz, 2, inst)
    call legion_index_launcher_execute_f(runtime, ctx, daxpy_launcher, daxpy_task_future_map)
    
    !create HDF5 file
    hdf5_file_is_valid = generate_hdf_file(hdf5_file_name, hdf5_dataset_name, num_elements)
    call legion_field_map_create_f(hdf5_field_map)
    call legion_field_map_insert_f(hdf5_field_map, 3, hdf5_dataset_name)
    call legion_runtime_attach_hdf5_f(runtime, ctx, hdf5_file_name, &
                                      cp_lr, cp_lr, hdf5_field_map, LEGION_FILE_READ_WRITE, cp_pr)
    
    ! create copy task
    call legion_copy_launcher_create_f(pred, 0, tag, cp_launcher)
    call legion_copy_launcher_add_src_region_requirement_lr_f(cp_launcher, output_lr, READ_ONLY, EXCLUSIVE, &
                                                              output_lr, tag, verified, rridx_cp)
    call legion_copy_launcher_add_dst_region_requirement_lr_f(cp_launcher, cp_lr, WRITE_DISCARD, EXCLUSIVE, &
                                                              cp_lr, tag, verified, rridx_cp)
    call legion_copy_launcher_add_src_field_f(cp_launcher, 0, 2, inst)
    call legion_copy_launcher_add_dst_field_f(cp_launcher, 0, 3, inst)
    call legion_copy_launcher_execute_f(runtime, ctx, cp_launcher)
    call legion_runtime_detach_hdf5_f(runtime, ctx, cp_pr)
    
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
    call legion_index_launcher_destroy_f(init_launcher_x)
    call legion_index_launcher_destroy_f(init_launcher_y)
    call legion_index_launcher_destroy_f(daxpy_launcher)
    call legion_copy_launcher_destroy_f(cp_launcher)
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
    integer(c_int) :: task_id_1, task_id_2, task_id_3, task_id_4
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
                                                                c_char_"cpu_variant"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
    
    config_options%leaf = .TRUE.
    c_func_ptr = c_funloc(init_task)

    task_id_2 = legion_runtime_preregister_task_variant_fnptr_f(INIT_TASK_ID, c_char_"init_task"//c_null_char, &
                                                                c_char_"cpu_variant"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
                                                                
    c_func_ptr = c_funloc(daxpy_task)

    task_id_3 = legion_runtime_preregister_task_variant_fnptr_f(DAXPY_TASK_ID, c_char_"daxpy_task"//c_null_char, &
                                                                c_char_"cpu_variant"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
    
    c_func_ptr = c_funloc(check_task)

    task_id_4 = legion_runtime_preregister_task_variant_fnptr_f(CHECK_TASK_ID, c_char_"check_task"//c_null_char, &
                                                                c_char_"cpu_variant"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
    runtime_start_rv = legion_runtime_start_f(0, c_null_ptr, background)
End Program Hello
