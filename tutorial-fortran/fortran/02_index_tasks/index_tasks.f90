module hello_world_index
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID = 0
  integer(c_int), parameter :: HELLO_WORLD_TASK_ID = 1
  
contains
  subroutine hello_world_task(tdata, tdatalen, userdata, userlen, p)
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
    integer(c_size_t) :: global_arglen, local_arglen
    integer*4, pointer :: global_task_args, local_task_args
    type(c_ptr) :: global_task_args_ptr, local_task_args_ptr
        
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    call legion_task_get_args_f(task, global_task_args_ptr)
    call c_f_pointer(global_task_args_ptr, global_task_args)
    call legion_task_get_arglen_f(task, global_arglen)
    
    call legion_task_get_local_args_f(task, local_task_args_ptr)
    call c_f_pointer(local_task_args_ptr, local_task_args)
    call legion_task_get_local_arglen_f(task, local_arglen)
    Print *, "Hello World Index Task!", local_task_args, local_arglen, global_task_args, global_arglen
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
  end subroutine hello_world_task

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
    
    type(legion_predicate_f_t) :: pred
    type(legion_task_argument_f_t) :: global_task_args, local_task_args(0:9)
    type(legion_index_launcher_f_t) :: index_launcher
    integer(c_long) :: tag = 0
    type(legion_argument_map_f_t) :: arg_map
    type(legion_point_1d_f_t) :: lo, hi, tmp_p
    type(legion_domain_point_f_t) :: dp
    type(legion_rect_1d_f_t) :: launch_bound
    type(legion_domain_f_t) :: domain
    
    type(legion_future_map_f_t) :: hello_world_task_future_map
    
    integer*4, target :: i = 0
    integer*4, target :: input = 0
    
    Print *, "TOP Level Task!"
    
    
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    call legion_predicate_true_f(pred) 
    global_task_args%args = c_loc(i)
    global_task_args%arglen = c_sizeof(i)
    
    ! init launch domain
    lo%x(0) = 0
    hi%x(0) = 9
    launch_bound%lo = lo
    launch_bound%hi = hi
    call legion_domain_from_rect_1d_f(launch_bound, domain)
    
    ! create arg map
    call legion_argument_map_create_f(arg_map)
    do i = 0, 9
      input = i + 10
      local_task_args(i)%args = c_loc(input)
      local_task_args(i)%arglen = c_sizeof(input)
      tmp_p%x(0) = i
      call legion_domain_point_from_point_1d_f(tmp_p, dp)
      call legion_argument_map_set_point_f(arg_map, dp, local_task_args(i), .TRUE.)
    end do
    
    ! index launcher
    call legion_index_launcher_create_f(HELLO_WORLD_TASK_ID, domain, global_task_args, &
                                        arg_map, pred, .FALSE., 0, tag, index_launcher)
    call legion_index_launcher_execute_f(runtime, ctx, index_launcher, hello_world_task_future_map)
    call legion_future_map_wait_all_results_f(hello_world_task_future_map)
    
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
  end subroutine top_level_task
end module hello_world_index

Program hello_index
    use legion_fortran
    use iso_c_binding
    use hello_world_index
    implicit none
    type(legion_execution_constraint_set_f_t) :: execution_constraints
    type(legion_task_layout_constraint_set_f_t) :: layout_constraints
    type(legion_task_config_options_f_t) :: config_options
    integer(c_int) :: task_id_1, task_id_2
    integer(c_size_t) :: userlen = 0
    integer(c_int) :: runtime_start_rv
    type(c_funptr) :: c_func_ptr
        
    Print *, "Hello World from Main!"
    call legion_runtime_set_top_level_task_id_f(TOP_LEVEL_TASK_ID)
    execution_constraints = legion_execution_constraint_set_create_f()
    call legion_execution_constraint_set_add_processor_constraint_f(execution_constraints, LOC_PROC)
    layout_constraints = legion_task_layout_constraint_set_create_f()
    config_options%leaf = .FALSE.
    config_options%inner = .FALSE.
    config_options%idempotent = .FALSE.
    
    c_func_ptr = c_funloc(top_level_task)
    
    task_id_1 = legion_runtime_preregister_task_variant_fnptr_f( &
      TOP_LEVEL_TASK_ID, c_char_"top_level_task"//c_null_char, &
      c_char_"cpu_variant"//c_null_char, &
      execution_constraints, &
      layout_constraints, &
      config_options, &
      c_func_ptr, &
      c_null_ptr, &
      userlen)
    
    c_func_ptr = c_funloc(hello_world_task)

    task_id_2 = legion_runtime_preregister_task_variant_fnptr_f( &
      HELLO_WORLD_TASK_ID, c_char_"hello_world_task"//c_null_char, &
      c_char_"cpu_variant"//c_null_char, &
      execution_constraints, &
      layout_constraints, &
      config_options, &
      c_func_ptr, &
      c_null_ptr, &
      userlen)
    
    runtime_start_rv = legion_runtime_start_f(0, c_null_ptr, .FALSE.)
End Program hello_index
