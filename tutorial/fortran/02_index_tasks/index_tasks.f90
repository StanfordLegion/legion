module index_space_test
  use legion_fortran
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID = 0
  integer(c_int), parameter :: INDEX_SPACE_TASK_ID = 1
  
contains
  subroutine index_space_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(FRuntime) :: runtime
    type(FContext) :: ctx
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    
    integer(c_size_t) :: arglen
    integer(kind=4), pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    integer(kind=4), target :: input
        
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    arglen = task%get_local_arglen()
    task_arg_ptr = task%get_local_args()
    call c_f_pointer(task_arg_ptr, task_arg)
    input = task_arg
    print *, "Index Task", input
    input = input * 2
    call legion_task_epilog(runtime, ctx, c_loc(input), c_sizeof(input))
  end subroutine index_space_task

  subroutine top_level_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(FRuntime) :: runtime
    type(FContext) :: ctx
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FArgumentMap) :: arg_map
    type(FIndexLauncher) :: index_launcher
    type(FRect1D) :: launch_bounds
    type(FIndexSpace) :: launch_is
    type(FFutureMap) :: fm
    type(FFuture) :: f
      
    integer :: num_points = 4
    
    integer :: i = 0
    integer(kind=4), target :: input = 0
    integer(kind=4) :: received, expected
    logical :: all_passed = .true.
    
    Print *, "TOP Level Task!"
    
    
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)

    launch_bounds = FRect1D(0, num_points-1)
    launch_is = runtime%create_index_space(ctx, launch_bounds)
    
    arg_map = FArgumentMap()
    do i = 0, num_points-1
      input = i + 10
      call arg_map%set_point(i, FTaskArgument(c_loc(input), c_sizeof(input)))
    end do
    
    index_launcher = FIndexLauncher(INDEX_SPACE_TASK_ID, launch_is, &
                                    FTaskArgument(), arg_map)
    fm = runtime%execute_index_space(ctx, index_launcher)
    call fm%wait_all_results()
    do i = 0, num_points-1
      f = fm%get_future(i)
      call f%get_result(received)
      print *, "result", received
      expected = 2*(i+10)
      if (expected .ne. received) then
        print *, "Check failed for point", i, expected, received
      end if
    end do
    
    if (all_passed) then
      print *, "All checks passed!"
    end if
    
    call index_launcher%destroy()
    call arg_map%destroy()
    
    call legion_task_epilog(runtime, ctx)
  end subroutine top_level_task
end module index_space_test

Program index_space_main
  use iso_c_binding
  use legion_fortran
  use index_space_test
  implicit none
  
  type(FTaskVariantRegistrar) :: registrar_top, registrar_index_space
      
  call set_top_level_task_id(TOP_LEVEL_TASK_ID)
  
  registrar_top = FTaskVariantRegistrar(TOP_LEVEL_TASK_ID)
  call registrar_top%add_constraint(FProcessorConstraint(LOC_PROC))
  call preregister_task_variant(top_level_task, registrar_top, "top_level_task")
  call registrar_top%destroy()
  
  registrar_index_space = FTaskVariantRegistrar(INDEX_SPACE_TASK_ID)
  call registrar_index_space%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_index_space%set_leaf()
  call preregister_task_variant(index_space_task, registrar_index_space, "index_space_task")
  call registrar_index_space%destroy()
  
  call legion_runtime_start()
End Program index_space_main
