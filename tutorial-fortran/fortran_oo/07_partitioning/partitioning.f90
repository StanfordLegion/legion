module daxpy
  use legion_fortran
  use legion_fortran_object_oriented
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID=0
  integer(c_int), parameter :: INIT_TASK_ID=1
  integer(c_int), parameter :: DAXPY_TASK_ID=2
  integer(c_int), parameter :: CHECK_TASK_ID=3
  
contains
  subroutine init_task(tdata, tdatalen, userdata, userlen, p)
    use legion_fortran_c_interface
    implicit none
    
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(FFieldAccessor1D) :: accessor
        
    real(kind=8), target :: x_value
    integer :: i, fid
    
    type(FContext) :: ctx
    type(FRuntime) :: runtime
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FPhysicalRegion) :: physical_region
    type(FLogicalRegion) :: logical_region
    type(FRegionRequirement) :: region_requirement
    type(FIndexSpace) :: index_space
    type(FRect1D) :: rect_1d
    type(FDomainPointIterator) :: pir
    type(FDomainPoint) :: dp
      
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    region_requirement = task%get_region_requirement(0)
    fid = region_requirement%get_privilege_field(0)
                                
    physical_region = pr_list%get_region(0)                         
    
    accessor = FFieldAccessor1D(physical_region, fid, c_sizeof(x_value))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_1d = runtime%get_index_space_domain(ctx, index_space)
    
    Print *, "Init Task!", fid
    
    pir = FDomainPointIterator(rect_1d)
    
    do while(pir%has_next() .eqv. .true.)
      dp = pir%step()
      call accessor%write_point(dp%get_point_1d(), x_value)
    end do
    
    call pir%destroy()
    
    call legion_task_epilog(runtime, ctx)
  end subroutine init_task

  subroutine daxpy_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p

    type(legion_rect_1d_f_t) :: index_rect
    real(kind=8), target :: z_value, x_value, y_value
    integer :: i
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
    
    accessor_x = FFieldAccessor1D(physical_region_0, 0, c_sizeof(x_value))
    accessor_y = FFieldAccessor1D(physical_region_0, 1, c_sizeof(y_value))
    accessor_z = FFieldAccessor1D(physical_region_1, 2, c_sizeof(z_value))
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
    
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p

    type(legion_rect_1d_f_t) :: index_rect
    real(kind=8), target :: x_value = 0
    real(kind=8), target :: y_value = 0
    real(kind=8), target :: z_value = 0
    integer :: i
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
    
    accessor_x = FFieldAccessor1D(physical_region_0, 0, c_sizeof(x_value))
    accessor_y = FFieldAccessor1D(physical_region_0, 1, c_sizeof(y_value))
    accessor_z = FFieldAccessor1D(physical_region_1, 2, c_sizeof(z_value))
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
    
    type(c_ptr), intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    real(kind=8) :: real_number = 0.0
    real(kind=8), target :: alpha = 0.1    

    integer :: num_elements = 1024
    integer :: num_subregions = 4
    
    type(FRect1D) :: elem_rect, color_bounds
    type(FRuntime) :: runtime
    type(FContext) :: ctx
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FIndexSpace) :: is, color_is
    type(FFieldSpace) :: input_fs, output_fs
    type(FFieldAllocator) :: ifs_allocator, ofs_allocator
    type(FIndexPartition) :: ip
    type(FLogicalRegion) :: input_lr, output_lr
    type(FLogicalPartition) :: input_lp, output_lp
    type(FTaskLauncher) :: check_launcher
    type(FIndexLauncher) :: init_launcher_x, init_launcher_y, daxpy_launcher
    type(FFuture) :: task_future
    type(FArgumentMap) :: arg_map
    type(FFutureMap) :: task_future_map
    
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
    
    ! create partition
    color_bounds = FRect1D(0, num_subregions-1)
    color_is = runtime%create_index_space(ctx, color_bounds)
    ip = runtime%create_equal_partition(ctx, is, color_is)
    input_lp = runtime%get_logical_partition(ctx, input_lr, ip)
    output_lp = runtime%get_logical_partition(ctx, output_lr, ip)
    
    !init task for X
    init_launcher_x = FIndexLauncher(INIT_TASK_ID, color_is, &
                                     FTaskArgument(), arg_map)
    call init_launcher_x%add_region_requirement(input_lp, 0, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              input_lr)                                          
    call init_launcher_x%add_field(0, 0)
    task_future_map = runtime%execute_index_space(ctx, init_launcher_x)
    
    !init task for Y
    init_launcher_y = FIndexLauncher(INIT_TASK_ID, color_is, &
                                     FTaskArgument(), arg_map)
    call init_launcher_y%add_region_requirement(input_lp, 0, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              input_lr)                                          
    call init_launcher_y%add_field(0, 1)
    task_future_map = runtime%execute_index_space(ctx, init_launcher_y)
    
    !daxpy task
    daxpy_launcher = FIndexLauncher(DAXPY_TASK_ID, color_is, &
                                   FTaskArgument(c_loc(alpha), c_sizeof(alpha)), arg_map)
    call daxpy_launcher%add_region_requirement(input_lp, 0, & 
                                              READ_ONLY, EXCLUSIVE, &
                                              input_lr)                                          
    call daxpy_launcher%add_field(0, 0)
    call daxpy_launcher%add_field(0, 1)
    call daxpy_launcher%add_region_requirement(output_lp, 0, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              output_lr)                                          
    call daxpy_launcher%add_field(1, 2)
    task_future_map= runtime%execute_index_space(ctx, daxpy_launcher)
    
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
  
  type(legion_execution_constraint_set_f_t) :: execution_constraints
  type(legion_task_layout_constraint_set_f_t) :: layout_constraints
  type(legion_task_config_options_f_t) :: config_options
  integer(c_int) :: task_id_1, task_id_2, task_id_3, task_id_4, variant_id
  integer(c_size_t) :: userlen = 0
  integer(c_int) :: runtime_start_rv
  type(c_funptr) :: c_func_ptr
      
  call legion_runtime_set_top_level_task_id_f(TOP_LEVEL_TASK_ID)
  call legion_execution_constraint_set_create_f(execution_constraints)
  call legion_execution_constraint_set_add_processor_constraint_f(execution_constraints, LOC_PROC)
  call legion_task_layout_constraint_set_create_f(layout_constraints)
  config_options%leaf = .false.
  config_options%inner = .false.
  config_options%idempotent = .false.
  
  c_func_ptr = c_funloc(top_level_task)
  
  variant_id = 1
  
  call legion_runtime_preregister_task_variant_fnptr_f(TOP_LEVEL_TASK_ID, variant_id, c_char_"top_level_task"//c_null_char, &
                                                      c_char_"cpu_variant"//c_null_char, &
                                                      execution_constraints, &
                                                      layout_constraints, &
                                                      config_options, &
                                                      c_func_ptr, &
                                                      c_null_ptr, &
                                                      userlen, task_id_1)
  
  c_func_ptr = c_funloc(init_task)

  call legion_runtime_preregister_task_variant_fnptr_f(INIT_TASK_ID, variant_id, c_char_"init_task"//c_null_char, &
                                                      c_char_"cpu_variant"//c_null_char, &
                                                      execution_constraints, &
                                                      layout_constraints, &
                                                      config_options, &
                                                      c_func_ptr, &
                                                      c_null_ptr, &
                                                      userlen, task_id_2)
                                                              
  c_func_ptr = c_funloc(daxpy_task)

  call legion_runtime_preregister_task_variant_fnptr_f(DAXPY_TASK_ID, variant_id, c_char_"daxpy_task"//c_null_char, &
                                                      c_char_"cpu_variant"//c_null_char, &
                                                      execution_constraints, &
                                                      layout_constraints, &
                                                      config_options, &
                                                      c_func_ptr, &
                                                      c_null_ptr, &
                                                      userlen, task_id_3)
  
  c_func_ptr = c_funloc(check_task)

  call legion_runtime_preregister_task_variant_fnptr_f(CHECK_TASK_ID, variant_id, c_char_"check_task"//c_null_char, &
                                                      c_char_"cpu_variant"//c_null_char, &
                                                      execution_constraints, &
                                                      layout_constraints, &
                                                      config_options, &
                                                      c_func_ptr, &
                                                      c_null_ptr, &
                                                      userlen, task_id_4)
  call legion_runtime_start_f(0, c_null_ptr, .false., runtime_start_rv)
End Program daxpy_1d_accessor
