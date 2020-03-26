module tasks_and_futures
  use legion_fortran
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID = 0
  integer(c_int), parameter :: FIBONACCI_TASK_ID = 1
  integer(c_int), parameter :: SUM_TASK_ID = 2
  
contains
  subroutine sum_task(tdata, tdatalen, userdata, userlen, p)
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
    type(FFuture) :: f1, f2
    integer(kind=4) :: r1, r2
    integer(kind=4), target :: sum    
        
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    f1 = task%get_future(0)
    call f1%get_result(r1)
    f2 = task%get_future(1)
    call f2%get_result(r2)
    sum = r1 + r2

    call legion_task_epilog(runtime, ctx, c_loc(sum), c_sizeof(sum))
  end subroutine sum_task
  
  subroutine fibonacci_task(tdata, tdatalen, userdata, userlen, p)
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
    type(FTaskLauncher) :: t1, t2, sum
    type(FFuture) :: f1, f2, result
    integer(c_size_t) :: arglen
    integer(kind=4), pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    integer(kind=4), target :: fib_num, fib1, fib2
    integer(kind=4), target :: ret_num
        
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    arglen = task%get_arglen()
    task_arg_ptr = task%get_args()
    call c_f_pointer(task_arg_ptr, task_arg)
    fib_num = task_arg
    
    if (fib_num == 0) then
      ret_num = 0
      call legion_task_epilog(runtime, ctx, c_loc(ret_num), c_sizeof(ret_num))
      return
    end if
    if (fib_num == 1) then
      ret_num = 1
      call legion_task_epilog(runtime, ctx, c_loc(ret_num), c_sizeof(ret_num))
      return
    end if
    
    fib1 = fib_num - 1
    t1 = FTaskLauncher(FIBONACCI_TASK_ID, FTaskArgument(c_loc(fib1), c_sizeof(fib1)))
    f1 = runtime%execute_task(ctx, t1)
    
    fib2 = fib_num - 2
    t2 = FTaskLauncher(FIBONACCI_TASK_ID, FTaskArgument(c_loc(fib2), c_sizeof(fib2)))
    f2 = runtime%execute_task(ctx, t2)

    sum = FTaskLauncher(SUM_TASK_ID, FTaskArgument())
    call sum%add_future(f1)
    call sum%add_future(f2)
    result = runtime%execute_task(ctx, sum)
    call result%get_result(ret_num)
    
    call legion_task_epilog(runtime, ctx, c_loc(ret_num), c_sizeof(ret_num))
  end subroutine fibonacci_task

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
    type(FTaskLauncher) :: launcher
    type(FFuture) :: future
    integer(kind=4), target :: num_fibonacci = 7
    integer(kind=4) :: result
    
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    launcher = FTaskLauncher(FIBONACCI_TASK_ID, FTaskArgument(c_loc(num_fibonacci), c_sizeof(num_fibonacci)))
    future = runtime%execute_task(ctx, launcher)
    call future%get_result(result)
    
    print *, "fib", num_fibonacci, result
    
    call legion_task_epilog(runtime, ctx)
  end subroutine top_level_task
end module tasks_and_futures

Program tasks_and_futures_main
  use iso_c_binding
  use legion_fortran
  use tasks_and_futures
  implicit none
  
  type(FTaskVariantRegistrar) :: registrar_top, registrar_fib, registrar_sum
      
  call set_top_level_task_id(TOP_LEVEL_TASK_ID)
  
  registrar_top = FTaskVariantRegistrar(TOP_LEVEL_TASK_ID)
  call registrar_top%add_constraint(FProcessorConstraint(LOC_PROC))
  call preregister_task_variant(top_level_task, registrar_top, "top_level_task")
  call registrar_top%destroy()
  
  registrar_fib = FTaskVariantRegistrar(FIBONACCI_TASK_ID)
  call registrar_fib%add_constraint(FProcessorConstraint(LOC_PROC))
  call preregister_task_variant(fibonacci_task, registrar_fib, "fibonacci_task")
  call registrar_fib%destroy()
  
  registrar_sum = FTaskVariantRegistrar(SUM_TASK_ID)
  call registrar_sum%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_sum%set_leaf()
  call preregister_task_variant(sum_task, registrar_sum, "sum_task")
  call registrar_sum%destroy()
  
  call legion_runtime_start()
End Program tasks_and_futures_main
