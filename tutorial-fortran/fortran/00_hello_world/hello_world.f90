module hello_world
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID = 0
  integer(c_int), parameter :: HELLO_WORLD_TASK_ID = 1
  
contains
  subroutine hello_world_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
    implicit none
  
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), optional, intent(in) ::userdata
    integer(c_size_t), value, optional, intent(in) :: userlen
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
      
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    call legion_task_get_args_f(task, task_arg_ptr)
    call c_f_pointer(task_arg_ptr, task_arg)
    call legion_task_get_arglen_f(task, arglen)
    Print *, "Hello World Task!", task_arg, arglen
  
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
  end subroutine hello_world_task

  subroutine top_level_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran
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
    type(legion_task_argument_f_t) :: task_args
    type(legion_task_launcher_f_t) :: launcher
    integer(c_long) :: tag = 0
    type(legion_future_f_t) :: hello_world_task_future
  
    integer*4, target :: i
  
    Print *, "TOP Level Task!"
  
    call legion_task_preamble_f(tdata, tdatalen, p, &
                                task, &
                                regionptr, num_regions, &
                                ctx, runtime)

    call legion_predicate_true_f(pred) 
    do i = 0, 10
      task_args%args = c_loc(i)
      task_args%arglen = c_sizeof(i)
      call legion_task_launcher_create_f(HELLO_WORLD_TASK_ID, task_args, pred, 0, tag, launcher)
      call legion_task_launcher_execute_f(runtime, ctx, launcher, hello_world_task_future)
    end do
    call legion_task_launcher_destroy_f(launcher)
    call legion_task_postamble_f(runtime, ctx, c_null_ptr, retsize)
  end subroutine top_level_task
end module hello_world
  

Program hello
  use legion_fortran
  use iso_c_binding
  use hello_world
  implicit none
  
  type(legion_execution_constraint_set_f_t) :: execution_constraints
  type(legion_task_layout_constraint_set_f_t) :: layout_constraints
  type(legion_task_config_options_f_t) :: config_options
  integer(c_int) :: task_id_1, task_id_2, runtime_start_rv
  integer(c_size_t) :: userlen = 0
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
  
  call legion_runtime_preregister_task_variant_fnptr_f( &
    TOP_LEVEL_TASK_ID, c_char_"top_level_task"//c_null_char, &
    c_char_"cpu_variant"//c_null_char, &
    execution_constraints, &
    layout_constraints, &
    config_options, &
    c_func_ptr, &
    c_null_ptr, &
    userlen, task_id_1)
  
  c_func_ptr = c_funloc(hello_world_task)

  call legion_runtime_preregister_task_variant_fnptr_f( &
    HELLO_WORLD_TASK_ID, c_char_"hello_world_task"//c_null_char, &
    c_char_"cpu_variant"//c_null_char, &
    execution_constraints, &
    layout_constraints, &
    config_options, &
    c_func_ptr, &
    c_null_ptr, &
    userlen, task_id_2)
  
  call legion_runtime_start_f(0, c_null_ptr, .false., runtime_start_rv)
End Program hello
