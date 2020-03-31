module daxpy
  use legion_fortran
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID=0
  integer(c_int), parameter :: INIT_TASK_ID=1
  integer(c_int), parameter :: DAXPY_TASK_ID=2
  integer(c_int), parameter :: CHECK_TASK_ID=3
  
contains
  subroutine init_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(FFieldAccessor1D) :: accessor
        
    type(legion_rect_1d_f_t) :: index_rect
    real(kind=8), target :: x_value
    type(FPoint1D) :: point_1d
    integer :: fid
    integer(kind=8) :: i
    
    type(FContext) :: ctx
    type(FRuntime) :: runtime
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FPhysicalRegion) :: physical_region
    type(FLogicalRegion) :: logical_region
    type(FRegionRequirement) :: region_requirement
    type(FIndexSpace) :: index_space
    type(FRect1D) :: rect_1d
      
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    region_requirement = task%get_region_requirement(0)
    fid = region_requirement%get_privilege_field(0)
                                
    physical_region = pr_list%get_region(0)                         
    
    accessor = FFieldAccessor1D(physical_region, fid, WRITE_DISCARD, c_sizeof(x_value))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_1d = runtime%get_index_space_domain(ctx, index_space)
    index_rect = rect_1d%rect
    
    Print *, "Init Task!", fid
    
    do i = index_rect%lo%x(0), index_rect%hi%x(0)
        point_1d = FPoint1D(i)
        x_value = 1.1 * (real(fid,8)+1) + real(i,8)
        call accessor%write_point(point_1d, x_value)
    end do
    
    call legion_task_epilog(runtime, ctx)
  end subroutine init_task

  subroutine daxpy_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p

    type(legion_rect_1d_f_t) :: index_rect
    real(kind=8), target :: z_value, x_value, y_value
    integer(kind=8) :: i
    integer(c_size_t) :: arglen
    real(kind=8), pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    real(kind=8) :: alpha
    
    type(FPoint1D) :: point_1d
    type(FFieldAccessor1D) :: accessor_x, accessor_y, accessor_z
    
    type(FContext) :: ctx
    type(FRuntime) :: runtime
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FPhysicalRegion) :: physical_region_0, physical_region_1
    type(FLogicalRegion) :: logical_region
    type(FRegionRequirement) :: region_requirement
    type(FIndexSpace) :: index_space
    type(FRect1D) :: rect_1d
      
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
                            
    arglen = task%get_arglen()
    task_arg_ptr = task%get_args()
    call c_f_pointer(task_arg_ptr, task_arg)
    alpha = task_arg
    
    region_requirement = task%get_region_requirement(0)
                                
    physical_region_0 = pr_list%get_region(0)     
    physical_region_1 = pr_list%get_region(1)  
    print *, "daxpy num_regions", pr_list%num_regions      
    
    accessor_x = FFieldAccessor1D(physical_region_0, 0, READ_ONLY, c_sizeof(x_value))
    accessor_y = FFieldAccessor1D(physical_region_0, 1, READ_ONLY, c_sizeof(y_value))
    accessor_z = FFieldAccessor1D(physical_region_1, 2, WRITE_DISCARD, c_sizeof(z_value))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_1d = runtime%get_index_space_domain(ctx, index_space)
    index_rect = rect_1d%rect
    
    Print *, "Daxpy Task!", alpha
    
    do i = index_rect%lo%x(0), index_rect%hi%x(0)
      point_1d = FPoint1D(i)
      call accessor_x%read_point(point_1d, x_value)
      call accessor_y%read_point(point_1d, y_value)
      z_value = alpha*x_value + y_value
      call accessor_z%write_point(point_1d, z_value)
    end do
    
    call legion_task_epilog(runtime, ctx)
  end subroutine daxpy_task

  subroutine check_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p

    type(legion_rect_1d_f_t) :: index_rect
    real(kind=8), target :: x_value = 0
    real(kind=8), target :: y_value = 0
    real(kind=8), target :: z_value = 0
    integer(kind=8) :: i
    logical :: all_passed = .true.
    integer(c_size_t) :: arglen
    real(kind=8), pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    real(kind=8) :: alpha
    
    type(FPoint1D) :: point_1d
    type(FFieldAccessor1D) :: accessor_x, accessor_y, accessor_z
    
    type(FContext) :: ctx
    type(FRuntime) :: runtime
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FPhysicalRegion) :: physical_region_0, physical_region_1
    type(FLogicalRegion) :: logical_region
    type(FRegionRequirement) :: region_requirement
    type(FIndexSpace) :: index_space
    type(FRect1D) :: rect_1d
      
    integer :: count
      
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    arglen = task%get_arglen()
    task_arg_ptr = task%get_args()
    call c_f_pointer(task_arg_ptr, task_arg)
    alpha = task_arg
    
    region_requirement = task%get_region_requirement(0)
                                
    physical_region_0 = pr_list%get_region(0)     
    physical_region_1 = pr_list%get_region(1)        
    
    accessor_x = FFieldAccessor1D(physical_region_0, 0, READ_ONLY, c_sizeof(x_value))
    accessor_y = FFieldAccessor1D(physical_region_0, 1, READ_ONLY, c_sizeof(y_value))
    accessor_z = FFieldAccessor1D(physical_region_1, 2, READ_ONLY, c_sizeof(z_value))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_1d = runtime%get_index_space_domain(ctx, index_space)
    index_rect = rect_1d%rect
    
    count = 0
    do i = index_rect%lo%x(0), index_rect%hi%x(0)
      count = count + 1
      point_1d = FPoint1D(i)
      call accessor_x%read_point(point_1d, x_value)
      call accessor_y%read_point(point_1d, y_value)
      call accessor_z%read_point(point_1d, z_value)
      if (alpha*x_value + y_value == z_value) then
      else
        print *, "wrong", i, x_value, y_value, z_value
        all_passed = .false.
      end if
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
    
    real(kind=8) :: real_number = 0.0
    real(kind=8), target :: alpha = 0.1    

    integer*4 :: num_elements = 1024
    
    type(FRect1D) :: elem_rect
    type(FRuntime) :: runtime
    type(FContext) :: ctx
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FIndexSpace) :: is
    type(FFieldSpace) :: input_fs, output_fs
    type(FFieldAllocator) :: ifs_allocator, ofs_allocator
    type(FLogicalRegion) :: input_lr, output_lr
    type(FTaskLauncher) :: init_launcher_x, init_launcher_y, daxpy_launcher, check_launcher
    type(FFuture) ::task_future
    
    Print *, "TOP Level Task!"
    
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    ! create index space, field space and logical region
    elem_rect = FRect1D(0, num_elements-1)
    is = runtime%create_index_space(ctx, elem_rect)
    
    input_fs = runtime%create_field_space(ctx)
    ifs_allocator = runtime%create_field_allocator(ctx, input_fs)
    call ifs_allocator%allocate_field(c_sizeof(real_number), 0)
    call ifs_allocator%allocate_field(c_sizeof(real_number), 1)
    call ifs_allocator%destroy()
    
    output_fs = runtime%create_field_space(ctx)
    ofs_allocator = runtime%create_field_allocator(ctx, output_fs)
    call ofs_allocator%allocate_field(c_sizeof(real_number), 2)
    call ofs_allocator%destroy()
    
    input_lr = runtime%create_logical_region(ctx, is, input_fs)
    output_lr = runtime%create_logical_region(ctx, is, output_fs)
    
    !init task for X
    init_launcher_x = FTaskLauncher(INIT_TASK_ID, FTaskArgument())
    call init_launcher_x%add_region_requirement(input_lr, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              input_lr)                                          
    call init_launcher_x%add_field(0, 0)
    task_future = runtime%execute_task(ctx, init_launcher_x)
    
    !init task for Y
    init_launcher_y = FTaskLauncher(INIT_TASK_ID, FTaskArgument())
    call init_launcher_y%add_region_requirement(input_lr, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              input_lr)                                          
    call init_launcher_y%add_field(0, 1)
    task_future = runtime%execute_task(ctx, init_launcher_y)
    
    !daxpy task
    daxpy_launcher = FTaskLauncher(DAXPY_TASK_ID, FTaskArgument(c_loc(alpha), c_sizeof(alpha)))
    call daxpy_launcher%add_region_requirement(input_lr, & 
                                              READ_ONLY, EXCLUSIVE, &
                                              input_lr)                                          
    call daxpy_launcher%add_field(0, 0)
    call daxpy_launcher%add_field(0, 1)
    call daxpy_launcher%add_region_requirement(output_lr, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              output_lr)                                          
    call daxpy_launcher%add_field(1, 2)
    task_future = runtime%execute_task(ctx, daxpy_launcher)
    
    !check task
    check_launcher = FTaskLauncher(CHECK_TASK_ID, FTaskArgument(c_loc(alpha), c_sizeof(alpha)))
    call check_launcher%add_region_requirement(input_lr, & 
                                              READ_ONLY, EXCLUSIVE, &
                                              input_lr)                                          
    call check_launcher%add_field(0, 0)
    call check_launcher%add_field(0, 1)
    call check_launcher%add_region_requirement(output_lr, & 
                                              READ_ONLY, EXCLUSIVE, &
                                              output_lr)                                          
    call check_launcher%add_field(1, 2)
    task_future = runtime%execute_task(ctx, check_launcher) 
    
    ! clean up
    call runtime%destroy_logical_region(ctx, input_lr)
    call runtime%destroy_logical_region(ctx, output_lr)
    call runtime%destroy_field_space(ctx, input_fs)
    call runtime%destroy_field_space(ctx, output_fs)
    call runtime%destroy_index_space(ctx, is)
    call init_launcher_x%destroy()
    call init_launcher_y%destroy()
    call daxpy_launcher%destroy()
    call check_launcher%destroy()
    
    call legion_task_epilog(runtime, ctx)
  end subroutine top_level_task
end module daxpy

Program daxpy_1d_accessor
  use legion_fortran
  use iso_c_binding
  use daxpy
  implicit none
  
  type(FTaskVariantRegistrar) :: registrar_top, registrar_init, registrar_daxpy, registrar_check
      
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
  
  registrar_daxpy = FTaskVariantRegistrar(DAXPY_TASK_ID)
  call registrar_daxpy%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_daxpy%set_leaf()
  call preregister_task_variant(daxpy_task, registrar_daxpy, "daxpy_task")
  call registrar_daxpy%destroy()
  
  registrar_check = FTaskVariantRegistrar(CHECK_TASK_ID)
  call registrar_check%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_check%set_leaf()
  call preregister_task_variant(check_task, registrar_check, "check_task")
  call registrar_check%destroy()
  
  call legion_runtime_start()
End Program daxpy_1d_accessor
