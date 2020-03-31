module physical_regions
  use legion_fortran
  use iso_c_binding
  implicit none
  
  integer(c_int), parameter :: TOP_LEVEL_TASK_ID = 0
  
contains
  subroutine top_level_task(tdata, tdatalen, userdata, userlen, p)
    implicit none
  
    type(c_ptr), value, intent(in) :: tdata
    integer(c_size_t), value, intent(in) :: tdatalen
    type(c_ptr), value, intent(in) ::userdata
    integer(c_size_t), value, intent(in) :: userlen
    integer(c_long_long), value, intent(in) :: p
    
    real(kind=8), target :: real_number = 0.0
    
    integer*4 :: num_elements = 1024
  
    type(FRuntime) :: runtime
    type(FContext) :: ctx
    type(FTask) :: task
    type(FPhysicalRegionList) :: pr_list
    type(FIndexSpace) :: is
    type(FRect1D) :: elem_rect
    type(FFieldSpace) :: input_fs
    type(FFieldAllocator) :: ifs_allocator
    type(FLogicalRegion) :: input_lr
    type(FInlineLauncher) :: input_launcher, output_launcher
    type(FPhysicalRegion) :: input_region, output_region
    type(FFieldAccessor1D) :: accessor_x, accessor_y
    type(FDomainPointIterator) :: pir
    type(FDomainPoint) :: dp
      
    real(kind=8), target :: x_value, y_value
    type(FPoint1D) :: point1d
    logical :: flag = .true.
    
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

    input_lr = runtime%create_logical_region(ctx, is, input_fs)
    
    call runtime%fill_field(ctx, input_lr, input_lr, 0, real_number)
    call runtime%fill_field(ctx, input_lr, input_lr, 1, real_number)
    
    input_launcher = FInlineLauncher(input_lr, READ_WRITE, EXCLUSIVE, input_lr)
    call input_launcher%add_field(0)
    call input_launcher%add_field(1)
    input_region = runtime%map_region(ctx, input_launcher)
    call input_region%wait_until_valid()
    
    accessor_x = FFieldAccessor1D(input_region, 0, WRITE_DISCARD, c_sizeof(x_value))
    accessor_y = FFieldAccessor1D(input_region, 1, WRITE_DISCARD, c_sizeof(y_value))
    
    pir = FDomainPointIterator(elem_rect)
    
    do while(pir%has_next() .eqv. .true.)
      dp = pir%step()
      point1d = dp%get_point_1d()
      x_value = real(point1d%point%x(0), 8) * 0.2 
      y_value = real(point1d%point%x(0), 8) * 0.1 
      call accessor_x%write_point(point1d, x_value)
      call accessor_y%write_point(point1d, y_value)
    end do
    call runtime%unmap_region(ctx, input_region)
    call input_launcher%destroy()
    
    output_launcher = FInlineLauncher(input_lr, READ_ONLY, EXCLUSIVE, input_lr)
    call output_launcher%add_field(0)
    call output_launcher%add_field(1)
    output_region = runtime%map_region(ctx, output_launcher)
    call output_region%wait_until_valid()
    
    accessor_x = FFieldAccessor1D(output_region, 0, READ_ONLY, c_sizeof(x_value))
    accessor_y = FFieldAccessor1D(output_region, 1, READ_ONLY, c_sizeof(y_value))
    
    pir = FDomainPointIterator(elem_rect)
    
    do while(pir%has_next() .eqv. .true.)
      dp = pir%step()
      point1d = dp%get_point_1d()
      call accessor_x%read_point(point1d, x_value)
      call accessor_y%read_point(point1d, y_value)
      if (x_value /= real(point1d%point%x(0), 8) * 0.2) then
        print *, "error", x_value 
        flag = .false.
      endif
      if (y_value /= real(point1d%point%x(0), 8) * 0.1) then
        print *, "error", y_value
        flag = .false. 
      endif
    end do
    call runtime%unmap_region(ctx, output_region)
    call output_launcher%destroy()
    
    if (flag) then
      print *, "Success"
    else
      print *, "Failed"
    endif
    
    ! clean up
    call runtime%destroy_logical_region(ctx, input_lr)
    call runtime%destroy_field_space(ctx, input_fs)
    call runtime%destroy_index_space(ctx, is)
    
    
    call legion_task_epilog(runtime, ctx)
  end subroutine top_level_task
end module physical_regions
  

Program physical_regions_main
  use legion_fortran
  use iso_c_binding
  use physical_regions
  implicit none
  
  type(FTaskVariantRegistrar) :: registrar_top
      
  call set_top_level_task_id(TOP_LEVEL_TASK_ID)
  
  registrar_top = FTaskVariantRegistrar(TOP_LEVEL_TASK_ID)
  call registrar_top%add_constraint(FProcessorConstraint(LOC_PROC))
  call preregister_task_variant(top_level_task, registrar_top, "top_level_task")
  call registrar_top%destroy()
  
  call legion_runtime_start()
End Program physical_regions_main
