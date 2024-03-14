module daxpy
  use legion_fortran
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID=0
  integer(c_int), parameter :: INIT_FIELD_TASK_ID=1
  integer(c_int), parameter :: STENCIL_TASK_ID=2
  integer(c_int), parameter :: CHECK_TASK_ID=3
  
contains
  subroutine init_field_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    type(FFieldAccessor1D) :: accessor
        
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
    type(FRect1D) :: rect_1d
    type(FDomainPointIterator) :: pir
    type(FDomainPoint) :: dp
      
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
    
    Print *, "Init Task!", fid
    
    pir = FDomainPointIterator(rect_1d)
    
    do while(pir%has_next() .eqv. .true.)
      dp = pir%step()
      x_value = 1.1
      call accessor%write_point(dp%get_point_1d(), x_value)
    end do
    
    call pir%destroy()
    
    call legion_task_epilog(runtime, ctx)
  end subroutine init_field_task

  subroutine stencil_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p

    real(kind=8), target :: l1, l2, r1, r2, result
    integer(c_size_t) :: arglen
    integer, pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    integer :: max_elements
    
    type(FFieldAccessor1D) :: read_acc, write_acc
    
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
    type(FPoint1D) :: lp1, lp2, rp1, rp2, cp
    integer(c_long_long) :: lo, hi
      
    integer :: count
      
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
                            
    arglen = task%get_arglen()
    task_arg_ptr = task%get_args()
    call c_f_pointer(task_arg_ptr, task_arg)
    max_elements = task_arg
    
    region_requirement = task%get_region_requirement(1)
                                
    physical_region_0 = pr_list%get_region(0)     
    physical_region_1 = pr_list%get_region(1)        
    
    read_acc = FFieldAccessor1D(physical_region_0, 0, READ_ONLY, c_sizeof(l1))
    write_acc = FFieldAccessor1D(physical_region_1, 1, WRITE_DISCARD, c_sizeof(l1))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_1d = runtime%get_index_space_domain(ctx, index_space)
    
    pir = FDomainPointIterator(rect_1d)
    
    count = 0
    lo = rect_1d%rect%lo%x(0)
    hi = rect_1d%rect%hi%x(0)
    if ((lo < 2) .or. (hi > (max_elements-3))) then
      do while(pir%has_next() .eqv. .true.)
        count = count + 1
        dp = pir%step()
        cp = dp%get_point_1d()
        if (cp%point%x(0) < 2) then
          lp2 = FPoint1D(0)
        else
          lp2 = FPoint1D(cp%point%x(0) - 2)
        endif
        call read_acc%read_point(lp2, l2)
        if (cp%point%x(0) < 1) then
          lp1 = FPoint1D(0)
        else
          lp1 = FPoint1D(cp%point%x(0) - 1)
        endif
        call read_acc%read_point(lp1, l1)
        if (cp%point%x(0) > (max_elements-2)) then
          rp1 = FPoint1D(max_elements-1)
        else
          rp1 = FPoint1D(cp%point%x(0) + 1)
        endif
        call read_acc%read_point(rp1, r1)
        if (cp%point%x(0) > (max_elements-3)) then
          rp2 = FPoint1D(max_elements-1)
        else
          rp2 = FPoint1D(cp%point%x(0) + 2)
        endif
        call read_acc%read_point(rp2, r2)
        result = l1 + l2 + r1 + r2
        call write_acc%write_point(cp, result) 
      end do
    else
      do while(pir%has_next() .eqv. .true.)
        count = count + 1
        dp = pir%step()
        cp = dp%get_point_1d()
        lp1 = FPoint1D(cp%point%x(0) - 1)
        lp2 = FPoint1D(cp%point%x(0) - 2)
        rp1 = FPoint1D(cp%point%x(0) + 1)
        rp2 = FPoint1D(cp%point%x(0) + 2)
        call read_acc%read_point(lp2, l2)
        call read_acc%read_point(lp1, l1)
        call read_acc%read_point(rp2, r1)
        call read_acc%read_point(rp2, r2)
        result = l1 + l2 + r1 + r2
        call write_acc%write_point(cp, result)
      end do
    endif
    
    Print *, "Stencil Task!", max_elements, count
    
    call pir%destroy()
    
    call legion_task_epilog(runtime, ctx)
  end subroutine stencil_task

  subroutine check_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
    
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p

    type(legion_rect_1d_f_t) :: index_rect
    real(kind=8), target :: received, expected, l1, l2, r1, r2
    logical :: all_passed = .true.
    integer(c_size_t) :: arglen
    integer, pointer :: task_arg
    type(c_ptr) :: task_arg_ptr
    integer :: max_elements
    
    type(FFieldAccessor1D) :: src_acc, dst_acc
    
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
    type(FPoint1D) :: lp1, lp2, rp1, rp2, cp
      
    integer :: count
      
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    arglen = task%get_arglen()
    task_arg_ptr = task%get_args()
    call c_f_pointer(task_arg_ptr, task_arg)
    max_elements = task_arg
    
    region_requirement = task%get_region_requirement(1)
                                
    physical_region_0 = pr_list%get_region(0)     
    physical_region_1 = pr_list%get_region(1)        
    
    src_acc = FFieldAccessor1D(physical_region_0, 0, READ_ONLY, c_sizeof(received))
    dst_acc = FFieldAccessor1D(physical_region_1, 1, READ_ONLY, c_sizeof(received))
    logical_region = region_requirement%get_region()
    index_space = logical_region%get_index_space()
    rect_1d = runtime%get_index_space_domain(ctx, index_space)
    index_rect = rect_1d%rect
    
    pir = FDomainPointIterator(rect_1d)
    
    count = 0

    do while(pir%has_next() .eqv. .true.)
      dp = pir%step()
      cp = dp%get_point_1d()
      if (cp%point%x(0) < 2) then
        lp2 = FPoint1D(0)
      else
        lp2 = FPoint1D(cp%point%x(0) - 2)
      endif
      call src_acc%read_point(lp2, l2)
      if (cp%point%x(0) < 1) then
        lp1 = FPoint1D(0)
      else
        lp1 = FPoint1D(cp%point%x(0) - 1)
      endif
      call src_acc%read_point(lp1, l1)
      if (cp%point%x(0) > (max_elements-2)) then
        rp1 = FPoint1D(max_elements-1)
      else
        rp1 = FPoint1D(cp%point%x(0) + 1)
      endif
      call src_acc%read_point(rp1, r1)
      if (cp%point%x(0) > (max_elements-3)) then
        rp2 = FPoint1D(max_elements-1)
      else
        rp2 = FPoint1D(cp%point%x(0) + 2)
      endif
      call src_acc%read_point(rp2, r2)
      expected = l1 + l2 + r1 + r2
      call dst_acc%read_point(cp, received) 
      count = count + 1
      if (expected .ne. received) then
        print *, "wrong", expected, received
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

    integer, target :: num_elements = 1024
    integer :: num_subregions = 4
    integer :: block_size
    
    type(FRect1D) :: elem_rect, color_bounds, extent
    type(FRuntime) :: runtime
    type(FContext) :: ctx
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FIndexSpace) :: is, color_is
    type(FFieldSpace) :: fs
    type(FFieldAllocator) :: fs_allocator
    type(FIndexPartition) :: disjoint_ip, ghost_ip
    type(FLogicalRegion) :: stencil_lr
    type(FLogicalPartition) :: disjoint_lp, ghost_lp
    type(FTaskLauncher) :: check_launcher
    type(FIndexLauncher) :: init_launcher, stencil_launcher
    type(FFuture) :: task_future
    type(FArgumentMap) :: arg_map
    type(FFutureMap) :: task_future_map
    type(FTransform1X1) :: transform
    
    Print *, "TOP Level Task!"
    
    call legion_task_prolog(tdata, tdatalen, userdata, userlen, p, &
                            task, pr_list, &
                            ctx, runtime)
    
    ! create index space, field space and logical region
    elem_rect = FRect1D(0, num_elements-1)
    is = runtime%create_index_space(ctx, elem_rect)
    
    fs = runtime%create_field_space(ctx)
    fs_allocator = runtime%create_field_allocator(ctx, fs)
    call fs_allocator%allocate_field(c_sizeof(real_number), 0)
    call fs_allocator%allocate_field(c_sizeof(real_number), 1)
    call fs_allocator%destroy()
    
    stencil_lr = runtime%create_logical_region(ctx, is, fs)
    
    ! create partition
    color_bounds = FRect1D(0, num_subregions-1)
    color_is = runtime%create_index_space(ctx, color_bounds)
    disjoint_ip = runtime%create_equal_partition(ctx, is, color_is)
    block_size = (num_elements + num_subregions - 1) / num_subregions
    transform%transform%trans = block_size
    extent = FRect1D(-2, block_size+1)
    ghost_ip = runtime%create_partition_by_restriction(ctx, is, color_is, transform, extent)
    
    disjoint_lp = runtime%get_logical_partition(stencil_lr, disjoint_ip)
    ghost_lp = runtime%get_logical_partition(stencil_lr, ghost_ip)
    
    arg_map = FArgumentMap()
    !init task
    init_launcher = FIndexLauncher(INIT_FIELD_TASK_ID, color_is, &
                                     FTaskArgument(), arg_map)
    call init_launcher%add_region_requirement(disjoint_lp, 0, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              stencil_lr)                                          
    call init_launcher%add_field(0, 0)
    task_future_map = runtime%execute_index_space(ctx, init_launcher)
    
    !stencil task
    stencil_launcher = FIndexLauncher(STENCIL_TASK_ID, color_is, &
                                   FTaskArgument(c_loc(num_elements), c_sizeof(num_elements)), arg_map)
    call stencil_launcher%add_region_requirement(ghost_lp, 0, & 
                                              READ_ONLY, EXCLUSIVE, &
                                              stencil_lr)                                          
    call stencil_launcher%add_field(0, 0)
    call stencil_launcher%add_region_requirement(disjoint_lp, 0, & 
                                              WRITE_DISCARD, EXCLUSIVE, &
                                              stencil_lr)                                          
    call stencil_launcher%add_field(1, 1)
    task_future_map= runtime%execute_index_space(ctx, stencil_launcher)
    
    !check task
    check_launcher = FTaskLauncher(CHECK_TASK_ID, FTaskArgument(c_loc(num_elements), c_sizeof(num_elements)))
    call check_launcher%add_region_requirement(stencil_lr, & 
                                              READ_ONLY, EXCLUSIVE, &
                                              stencil_lr)                                          
    call check_launcher%add_field(0, 0)
    call check_launcher%add_region_requirement(stencil_lr, & 
                                              READ_ONLY, EXCLUSIVE, &
                                              stencil_lr)                                          
    call check_launcher%add_field(1, 1)
    task_future = runtime%execute_task(ctx, check_launcher) 
    
    call arg_map%destroy()
    
    ! clean up
    call runtime%destroy_logical_region(ctx, stencil_lr)
    call runtime%destroy_field_space(ctx, fs)
    call runtime%destroy_index_space(ctx, is)
    call init_launcher%destroy()
    call stencil_launcher%destroy()
    call check_launcher%destroy()
    

    
    call legion_task_epilog(runtime, ctx)
  end subroutine top_level_task
end module daxpy

Program daxpy_1d_accessor
  use iso_c_binding
  use legion_fortran
  use daxpy
  implicit none
  
  type(FTaskVariantRegistrar) :: registrar_top, registrar_init, registrar_stencil, registrar_check
      
  call set_top_level_task_id(TOP_LEVEL_TASK_ID)
  
  registrar_top = FTaskVariantRegistrar(TOP_LEVEL_TASK_ID)
  call registrar_top%add_constraint(FProcessorConstraint(LOC_PROC))
  call preregister_task_variant(top_level_task, registrar_top, "top_level_task")
  call registrar_top%destroy()
  
  registrar_init = FTaskVariantRegistrar(INIT_FIELD_TASK_ID)
  call registrar_init%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_init%set_leaf()
  call preregister_task_variant(init_field_task, registrar_init, "init_field_task")
  call registrar_init%destroy()
  
  registrar_stencil = FTaskVariantRegistrar(STENCIL_TASK_ID)
  call registrar_stencil%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_stencil%set_leaf()
  call preregister_task_variant(stencil_task, registrar_stencil, "stencil_task")
  call registrar_stencil%destroy()
  
  registrar_check = FTaskVariantRegistrar(CHECK_TASK_ID)
  call registrar_check%add_constraint(FProcessorConstraint(LOC_PROC))
  call registrar_check%set_leaf()
  call preregister_task_variant(check_task, registrar_check, "check_task")
  call registrar_check%destroy()
  
  call legion_runtime_start()
End Program daxpy_1d_accessor
