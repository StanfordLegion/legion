function hello_world_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    use iso_c_binding
    implicit none
    
    integer(c_int) ::hello_world_task
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(legion_task_f_t) :: task
    integer(c_int) :: num_regions
    type(legion_context_f_t) :: ctx
    type(legion_runtime_f_t) :: runtime
    type(legion_physical_region_f_t) :: regionptr
    integer(c_size_t) :: retsize = 0
    integer(c_size_t) :: global_arglen, local_arglen
    integer*4, pointer :: global_task_args, local_task_args
    type(c_ptr) :: global_task_args_ptr, local_task_args_ptr
        
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    global_task_args_ptr = legion_task_get_args_f(task)
    call c_f_pointer(global_task_args_ptr, global_task_args)
    global_arglen = legion_task_get_arglen_f(task)
    
    local_task_args_ptr = legion_task_get_local_args_f(task)
    call c_f_pointer(local_task_args_ptr, local_task_args)
    local_arglen = legion_task_get_local_arglen_f(task)
    Print *, "Hello World Task!", local_task_args, local_arglen, global_task_args, global_arglen
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
    hello_world_task = 0
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
    type(legion_physical_region_f_t) :: regionptr
    integer(c_size_t) :: retsize = 0
    
    type(legion_predicate_f_t) :: pred
    type(legion_task_argument_f_t) :: global_task_args, local_task_args(0:9)
    type(legion_index_launcher_f_t) :: index_launcher
    integer(c_long) :: tag = 0
    type(legion_argument_map_f_t) :: arg_map
    type(legion_point_1d_f_t) :: lo, hi, tmp_p
    type(legion_domain_point_f_t) :: dp
    type(legion_rect_1d_f_t) :: launch_bound
    type(legion_domain_f_t) :: domain
    logical(c_bool) :: must = .FALSE.
    logical(c_bool) :: replace = .TRUE.
    
    type(legion_future_map_f_t) :: hello_world_task_future_map
    
    integer(c_int) :: HELLO_WORLD_TASK_ID=1
    integer*4, target :: i = 0
    integer*4, target :: input = 0
   ! common HELLO_WORLD_TASK_ID
    
    Print *, "TOP Level Task!"
    
    
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    pred = legion_predicate_true_f() 
    global_task_args%args = c_loc(i)
    global_task_args%arglen = c_sizeof(i)
    
    ! init launch domain
    lo%x(0) = 0
    hi%x(0) = 9
    launch_bound%lo = lo
    launch_bound%hi = hi
    domain = legion_domain_from_rect_1d_f(launch_bound)
    
    ! create arg map
    arg_map = legion_argument_map_create_f()
    do i = 0, 9
        input = i + 10
        local_task_args(i)%args = c_loc(input)
        local_task_args(i)%arglen = c_sizeof(input)
        tmp_p%x(0) = i
        dp = legion_domain_point_from_point_1d_f(tmp_p)
        call legion_argument_map_set_point_f(arg_map, dp, local_task_args(i), replace)
    end do
    
    ! index launcher
    index_launcher = legion_index_launcher_create_f(HELLO_WORLD_TASK_ID, domain, global_task_args, arg_map, pred, must, 0, tag)
    hello_world_task_future_map = legion_index_launcher_execute_f(runtime, ctx, index_launcher)
    call legion_future_map_wait_all_results_f(hello_world_task_future_map)
    
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
    integer(c_int) :: TOP_LEVEL_TASK_ID
    integer(c_int) :: HELLO_WORLD_TASK_ID
    integer(c_int) :: task_id_1, task_id_2
    integer(c_size_t) :: userlen = 0
    integer(c_int) :: runtime_start_rv
    logical(c_bool) :: background = .FALSE.
    type(c_funptr) :: c_func_ptr
    
    external top_level_task
    external hello_world_task
    
   ! common TOP_LEVEL_TASK_ID
    !common HELLO_WORLD_TASK_ID
    TOP_LEVEL_TASK_ID = 0
    HELLO_WORLD_TASK_ID = 1
        
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
    
    c_func_ptr = c_funloc(hello_world_task)

    task_id_2 = legion_runtime_preregister_task_variant_fnptr_f(HELLO_WORLD_TASK_ID, c_char_"hello_world_task"//c_null_char, &
                                                                execution_constraints, &
                                                                layout_constraints, &
                                                                config_options, &
                                                                c_func_ptr, &
                                                                c_null_ptr, &
                                                                userlen)
    
    runtime_start_rv = legion_runtime_start_f(0, c_null_ptr, background)
End Program Hello