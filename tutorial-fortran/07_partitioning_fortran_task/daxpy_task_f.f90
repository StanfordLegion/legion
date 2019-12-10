subroutine daxpy_task_f(task_c, regionptr_c, num_regions, ctx_c, runtime_c) bind(C)
    use iso_c_binding
    use legion_fortran
    implicit none
    
    type(legion_task_f_t), value, intent(in) :: task_c
    type(c_ptr), value, intent(in) ::regionptr_c
    integer(c_int), value, intent(in) :: num_regions
    type(legion_context_f_t), value, intent(in) :: ctx_c
    type(legion_runtime_f_t), value, intent(in) :: runtime_c

    real(kind=8), target :: z_value, x_value, y_value
    integer(c_size_t) :: arglen
    real(kind=8), pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    real(kind=8) :: alpha
    
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
    type(FDomainPointIterator) :: pir
    type(FDomainPoint) :: dp
    
    task%task = task_c
    pr_list%region_ptr = regionptr_c
    pr_list%num_regions = num_regions
    ctx%context = ctx_c
    runtime%runtime = runtime_c 
                            
    arglen = task%get_arglen()
    task_arg_ptr = task%get_args()
    call c_f_pointer(task_arg_ptr, task_arg)
    alpha = task_arg
    
    region_requirement = task%get_region_requirement(0)
                                
    physical_region_0 = pr_list%get_region(0)     
    physical_region_1 = pr_list%get_region(1)        
    
    accessor_x = FFieldAccessor1D(physical_region_0, 0, READ_ONLY, c_sizeof(x_value))
    accessor_y = FFieldAccessor1D(physical_region_0, 1, READ_ONLY, c_sizeof(y_value))
    accessor_z = FFieldAccessor1D(physical_region_1, 2, WRITE_DISCARD, c_sizeof(z_value))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_1d = runtime%get_index_space_domain(ctx, index_space)
    
    Print *, "Daxpy Task!", alpha
    
    pir = FDomainPointIterator(rect_1d)
    
    do while(pir%has_next() .eqv. .true.)
      dp = pir%step()
      call accessor_x%read_point(dp%get_point_1d(), x_value)
      call accessor_y%read_point(dp%get_point_1d(), y_value)
      z_value = alpha*x_value + y_value
      call accessor_z%write_point(dp%get_point_1d(), z_value)
    end do
    
    call pir%destroy()    
end subroutine daxpy_task_f

subroutine daxpy_task_f2(tdata, tdatalen, userdata, userlen, p) bind(C)
  use iso_c_binding
  use legion_fortran
  implicit none
  
  type(c_ptr), value, intent(in) :: tdata
  integer(c_size_t), value, intent(in) :: tdatalen
  type(c_ptr), value, intent(in) ::userdata
  integer(c_size_t), value, intent(in) :: userlen
  integer(c_long_long), value, intent(in) :: p

  real(kind=8), target :: z_value, x_value, y_value
  integer(c_size_t) :: arglen
  real(kind=8), pointer :: task_arg
  type(c_ptr) :: task_arg_ptr
  real(kind=8) :: alpha
  
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
  type(FDomainPointIterator) :: pir
  type(FDomainPoint) :: dp
    
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
  accessor_z = FFieldAccessor1D(physical_region_1, 2, WRITE_DISCARD, c_sizeof(z_value))
  logical_region = region_requirement%get_region()
  index_space = logical_region%get_index_space()
  rect_1d = runtime%get_index_space_domain(ctx, index_space)
  
  Print *, "Daxpy Task!", alpha
  
  pir = FDomainPointIterator(rect_1d)
  
  do while(pir%has_next() .eqv. .true.)
    dp = pir%step()
    call accessor_x%read_point(dp%get_point_1d(), x_value)
    call accessor_y%read_point(dp%get_point_1d(), y_value)
    z_value = alpha*x_value + y_value
    call accessor_z%write_point(dp%get_point_1d(), z_value)
  end do
  
  call pir%destroy()
  
  call legion_task_epilog(runtime, ctx)
end subroutine daxpy_task_f2

