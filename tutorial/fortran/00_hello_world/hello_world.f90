module hello_world
  use legion_fortran
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: HELLO_WORLD_TASK_ID = 0
  
contains
  subroutine hello_world_task(tdata, tdatalen, userdata, userlen, p)
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
    
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)

    Print *, "Hello World!"
    
    call legion_task_epilog(runtime, ctx)
  end subroutine hello_world_task
end module hello_world
  

Program hello_world_main
  use legion_fortran
  use iso_c_binding
  use hello_world
  implicit none
  
  type(FTaskVariantRegistrar) :: registrar_hello
      
  call set_top_level_task_id(HELLO_WORLD_TASK_ID)
  
  registrar_hello = FTaskVariantRegistrar(HELLO_WORLD_TASK_ID)
  call registrar_hello%add_constraint(FProcessorConstraint(LOC_PROC))
  call preregister_task_variant(hello_world_task, registrar_hello, "hello_world_task")
  call registrar_hello%destroy()
  
  call legion_runtime_start()
End Program hello_world_main
