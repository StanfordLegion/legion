subroutine daxpy_task_f(tdata, tdatalen, userdata, userlen, p) bind(C)
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
    real(kind=8), pointer :: task_arg
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
    call legion_physical_region_get_field_accessor_array_1d_f(pr1, 1, accessor_y)
    call legion_physical_region_get_field_accessor_array_1d_f(pr2, 2, accessor_z)
    call legion_task_get_index_space_from_logical_region_f(task, 0, index_space)
    call legion_index_space_get_domain_f(runtime, index_space, index_domain)
    call legion_domain_get_rect_1d_f(index_domain, index_rect)
    
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
end subroutine