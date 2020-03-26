module region_2d
  use legion_fortran
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID=0
  integer(c_int), parameter :: INIT_TASK_ID=1
  integer(c_int), parameter :: CHECK_TASK_ID=2
  
contains
  subroutine init_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran_c_interface
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(FFieldAccessor2D) :: accessor
        
    real(kind=8), target :: x_value
    integer :: fid
    
    type(FContext) :: ctx
    type(FRuntime) :: runtime
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FPhysicalRegion) :: physical_region
    type(FLogicalRegion) :: logical_region
    type(FRegionRequirement) :: region_requirement
    type(FIndexSpace) :: index_space
    type(FRect2D) :: rect_2d
    type(FDomainPointIterator) :: pir
    type(FDomainPoint) :: dp
      
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    region_requirement = task%get_region_requirement(0)
    fid = region_requirement%get_privilege_field(0)
                                
    physical_region = pr_list%get_region(0)                         
    
    accessor = FFieldAccessor2D(physical_region, fid, WRITE_DISCARD, c_sizeof(x_value))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_2d = runtime%get_index_space_domain(ctx, index_space)
    
    Print *, "Init Task!", fid
    
    pir = FDomainPointIterator(rect_2d)
    
    do while(pir%has_next() .eqv. .true.)
      dp = pir%step()
      x_value = 1.1
      call accessor%write_point(dp%get_point_2d(), x_value)
    end do
    
    call pir%destroy()
    
    call legion_task_epilog(runtime, ctx)
  end subroutine init_task

  subroutine check_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p

    type(legion_rect_2d_f_t) :: index_rect
    real(kind=8), target :: x_value = 0
    integer(kind=8) :: i, j
    logical :: all_passed = .true.
    
    type(FPoint2D) :: point_2d
    type(FFieldAccessor2D) :: accessor_x
    
    type(FContext) :: ctx
    type(FRuntime) :: runtime
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FPhysicalRegion) :: physical_region_0
    type(FLogicalRegion) :: logical_region
    type(FRegionRequirement) :: region_requirement
    type(FIndexSpace) :: index_space
    type(FRect2D) :: rect_2d
      
    integer :: count
      
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    region_requirement = task%get_region_requirement(0)
                                
    physical_region_0 = pr_list%get_region(0)          
    
    accessor_x = FFieldAccessor2D(physical_region_0, 0, READ_ONLY, c_sizeof(x_value))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_2d = runtime%get_index_space_domain(ctx, index_space)
    index_rect = rect_2d%rect
    
    count = 0
    do i = index_rect%lo%x(0), index_rect%hi%x(0)
      do j = index_rect%lo%x(0), index_rect%hi%x(0)
        count = count + 1
        point_2d = FPoint2D(i, j)
        call accessor_x%read_point(point_2d, x_value)
        if (x_value == 1.1) then
        else
          print *, "wrong", i, x_value
          all_passed = .false.
        end if
      end do
    end do
    
    if (all_passed .eqv. .true.) then
      print *, "Pass", count
    else
      print *, "Failed"
    end if
    
    call legion_task_epilog(runtime, ctx)
  end subroutine check_task

  subroutine top_level_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p 

    integer :: num_elements = 32
    real(kind=8) :: real_number = 0.0
    
    type(FPoint2D) :: point_lo, point_hi
    type(FRect2D) :: elem_rect
    type(FRuntime) :: runtime
    type(FContext) :: ctx
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FIndexSpace) :: is
    type(FFieldSpace) :: input_fs
    type(FFieldAllocator) :: ifs_allocator
    type(FLogicalRegion) :: input_lr
    type(FTaskLauncher) :: init_launcher_x, check_launcher
    type(FFuture) :: task_future
    
    Print *, "TOP Level Task!"
    
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    ! create index space, field space and logical region
    point_lo = FPoint2D(0,0)
    point_hi = FPoint2D(num_elements-1, num_elements-1)
    elem_rect = FRect2D(point_lo, point_hi)
    is = runtime%create_index_space(ctx, elem_rect)
    
    input_fs = runtime%create_field_space(ctx)
    ifs_allocator = runtime%create_field_allocator(ctx, input_fs)
    call ifs_allocator%allocate_field(c_sizeof(real_number), 0)
    call ifs_allocator%destroy()
    
    input_lr = runtime%create_logical_region(ctx, is, input_fs)
    
    !init task for X
    init_launcher_x = FTaskLauncher(INIT_TASK_ID, FTaskArgument())
    call init_launcher_x%add_region_requirement(input_lr, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              input_lr)                                          
    call init_launcher_x%add_field(0, 0)
    task_future = runtime%execute_task(ctx, init_launcher_x)
    
    !check task
    check_launcher = FTaskLauncher(CHECK_TASK_ID, FTaskArgument())
    call check_launcher%add_region_requirement(input_lr, & 
                                              READ_ONLY, EXCLUSIVE, &
                                              input_lr)                                          
    call check_launcher%add_field(0, 0)
    task_future = runtime%execute_task(ctx, check_launcher) 
    
    ! clean up
    call runtime%destroy_logical_region(ctx, input_lr)
    call runtime%destroy_field_space(ctx, input_fs)
    call runtime%destroy_index_space(ctx, is)
    call init_launcher_x%destroy()
    call check_launcher%destroy()
    
    call legion_task_epilog(runtime, ctx)
  end subroutine top_level_task
end module region_2d

Program daxpy_1d_accessor
  use iso_c_binding
  use legion_fortran
  use region_2d
  implicit none
  
  type(FTaskVariantRegistrar) :: registrar_top, registrar_init, registrar_check
      
  call set_top_level_task_id(TOP_LEVEL_TASK_ID)
  
  registrar_top = FTaskVariantRegistrar(TOP_LEVEL_TASK_ID)
  call registrar_top%add_constraint(FProcessorConstraint(LOC_PROC))
  call preregister_task_variant(top_level_task, registrar_top, "top_level_task")
  call registrar_top%destroy()
  
  registrar_init = FTaskVariantRegistrar(INIT_TASK_ID)
  call registrar_init%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_init%set_leaf()
  call preregister_task_variant(init_task, registrar_init, "init_task")
  call registrar_init%destroy()
  
  registrar_check = FTaskVariantRegistrar(CHECK_TASK_ID)
  call registrar_check%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_check%set_leaf()
  call preregister_task_variant(check_task, registrar_check, "check_task")
  call registrar_check%destroy()
  
  call legion_runtime_start()
End Program daxpy_1d_accessor
