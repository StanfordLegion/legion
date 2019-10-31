module legion_fortran_object_oriented
  use, intrinsic :: iso_c_binding
  use legion_fortran_types
  use legion_fortran_c_interface
  implicit none
  
  ! Point Class
  type FPoint
    integer :: dim
  end type FPoint

  type, extends(FPoint) :: FPoint1D
    type(legion_point_1d_f_t) :: point
  end type FPoint1D
  
  type, extends(FPoint) :: FPoint2D
    type(legion_point_2d_f_t) :: point
  end type FPoint2D
  
  type, extends(FPoint) :: FPoint3D
    type(legion_point_3d_f_t) :: point
  end type FPoint3D
  
  interface FPoint1D
    module procedure legion_point_1d_constructor_integer4
    module procedure legion_point_1d_constructor_integer8
  end interface
  
  interface FPoint2D
    module procedure legion_point_2d_constructor_integer4
    module procedure legion_point_2d_constructor_integer8
  end interface
  
  interface FPoint3D
    module procedure legion_point_3d_constructor_integer4
    module procedure legion_point_3d_constructor_integer8
  end interface
  
  ! Accessor Class
  type FFieldAccessor
    integer :: dim
    integer(c_size_t) :: data_size
  contains
    procedure :: init => legion_field_accessor_init
    procedure, private :: legion_field_accessor_read_point_ptr
    procedure, private :: legion_field_accessor_read_point_integer4
    procedure, private :: legion_field_accessor_read_point_integer8
    procedure, private :: legion_field_accessor_read_point_real4
    procedure, private :: legion_field_accessor_read_point_real8
    procedure, private :: legion_field_accessor_read_point_complex4
    procedure, private :: legion_field_accessor_read_point_complex8
    procedure, private :: legion_field_accessor_write_point_ptr
    procedure, private :: legion_field_accessor_write_point_integer4
    procedure, private :: legion_field_accessor_write_point_integer8
    procedure, private :: legion_field_accessor_write_point_real4
    procedure, private :: legion_field_accessor_write_point_real8
    procedure, private :: legion_field_accessor_write_point_complex4
    procedure, private :: legion_field_accessor_write_point_complex8
    generic :: read_point => legion_field_accessor_read_point_ptr, &
                             legion_field_accessor_read_point_integer4, legion_field_accessor_read_point_integer8, &
                             legion_field_accessor_read_point_real4, legion_field_accessor_read_point_real8, &
                             legion_field_accessor_read_point_complex4, legion_field_accessor_read_point_complex8
    generic :: write_point => legion_field_accessor_write_point_ptr, &
                              legion_field_accessor_write_point_integer4, legion_field_accessor_write_point_integer8, &
                              legion_field_accessor_write_point_real4, legion_field_accessor_write_point_real8, &
                              legion_field_accessor_write_point_complex4, legion_field_accessor_write_point_complex8
  end type FFieldAccessor

  type, extends(FFieldAccessor) :: FFieldAccessor1D
    type(legion_accessor_array_1d_f_t) :: accessor
  end type FFieldAccessor1D
  
  type, extends(FFieldAccessor) :: FFieldAccessor2D
    type(legion_accessor_array_2d_f_t) :: accessor
  end type FFieldAccessor2D
  
  type, extends(FFieldAccessor) :: FFieldAccessor3D
    type(legion_accessor_array_3d_f_t) :: accessor
  end type FFieldAccessor3D
  
  interface FFieldAccessor1D
    module procedure legion_field_accessor_1d_constructor
  end interface
  
  interface FFieldAccessor2D
    module procedure legion_field_accessor_2d_constructor
  end interface
  
  interface FFieldAccessor3D
    module procedure legion_field_accessor_3d_constructor
  end interface
  
  ! Domain Class
  type FDomain
    type(legion_domain_f_t) :: domain
  end type FDomain
  
  type, extends(FDomain) :: FRect1D
    type(legion_rect_1d_f_t) :: rect
  end type FRect1D
  
  type, extends(FDomain) :: FRect2D
    type(legion_rect_2d_f_t) :: rect
  end type FRect2D
  
  type, extends(FDomain) :: FRect3D
    type(legion_rect_3d_f_t) :: rect
  end type FRect3D
  
  ! IndexSpace Class
  type FIndexSpace
    type(legion_index_space_f_t) :: is
  end type FIndexSpace
  
  ! LogicalRegion Class
  type FLogicalRegion
    type(legion_logical_region_f_t) :: lr
  contains
    procedure :: get_index_space => legion_logical_region_get_index_space
  end type FLogicalRegion
  
  ! LogicalPartition Class
  type FLogicalPartition
    type(legion_logical_partition_f_t) :: lp
  end type FLogicalPartition
  
  ! PhysicalRegion
  type FPhysicalRegion
    type(legion_physical_region_f_t) :: pr
  end type FPhysicalRegion
  
  ! PhysicalRegionList Class
  type FPhysicalRegionList
    type(c_ptr) :: region_ptr
    integer :: num_regions
  contains
    procedure :: size => legion_physical_region_list_size
    procedure :: get_region => legion_physical_region_list_get_region_by_id
  end type FPhysicalRegionList
  
  ! RegionRequirement Class
  type FRegionRequirement
    type(legion_region_requirement_f_t) :: rr
  contains
    procedure :: get_region => legion_region_requirement_get_logical_region
  end type FRegionRequirement
  
  type FTask
    type(legion_task_f_t) :: task
  contains
    procedure :: get_region_requirement => legion_task_get_region_requirement_by_id
  end type FTask
  
  type FRuntime
    type(legion_runtime_f_t) :: runtime
  contains
    procedure, private :: legion_runtime_get_index_domain_return_domain
    procedure, private :: legion_runtime_get_index_domain_return_rect1
    generic :: get_index_space_domain => legion_runtime_get_index_domain_return_domain, &
                                         legion_runtime_get_index_domain_return_rect1
  end type FRuntime
  
  type FContext
    type(legion_context_f_t) :: context
  end type FContext
  
  Type CellReal8
    real(kind=8), dimension(:), pointer :: y
  end type CellReal8

  type LegionArray2DReal8
    type(CellReal8), dimension(3) :: x
    integer :: dim_x
    integer :: dim_y
    integer :: ld
  end type LegionArray2DReal8
  
  interface LegionArray2DReal8
    module procedure legion_array_2d_real8_constructor
  end interface
  
contains
  
  function legion_point_1d_constructor_integer4(x)
    implicit none
    
    type(FPoint1D)              :: legion_point_1d_constructor_integer4
    integer(kind=4), intent(in) :: x
    
    legion_point_1d_constructor_integer4%point%x(0) = int(x, 8)
      
  end function legion_point_1d_constructor_integer4
  
  function legion_point_1d_constructor_integer8(x)
    implicit none
    
    type(FPoint1D)              :: legion_point_1d_constructor_integer8
    integer(kind=8), intent(in) :: x
    
    legion_point_1d_constructor_integer8%point%x(0) = x
      
  end function legion_point_1d_constructor_integer8
  
  function legion_point_2d_constructor_integer4(x, y)
    implicit none
    
    type(FPoint2D)              :: legion_point_2d_constructor_integer4
    integer(kind=4), intent(in) :: x
    integer(kind=4), intent(in) :: y
    
    legion_point_2d_constructor_integer4%point%x(0) = int(x, 8)
    legion_point_2d_constructor_integer4%point%x(1) = int(y, 8)
      
  end function legion_point_2d_constructor_integer4
  
  function legion_point_2d_constructor_integer8(x, y)
    implicit none
    
    type(FPoint2D)              :: legion_point_2d_constructor_integer8
    integer(kind=8), intent(in) :: x
    integer(kind=8), intent(in) :: y
    
    legion_point_2d_constructor_integer8%point%x(0) = x
    legion_point_2d_constructor_integer8%point%x(1) = y
      
  end function legion_point_2d_constructor_integer8
  
  function legion_point_3d_constructor_integer4(x, y, z)
    implicit none
    
    type(FPoint3D)              :: legion_point_3d_constructor_integer4
    integer(kind=4), intent(in) :: x
    integer(kind=4), intent(in) :: y
    integer(kind=4), intent(in) :: z
    
    legion_point_3d_constructor_integer4%point%x(0) = int(x, 8)
    legion_point_3d_constructor_integer4%point%x(1) = int(y, 8)
    legion_point_3d_constructor_integer4%point%x(2) = int(z, 8)
      
  end function legion_point_3d_constructor_integer4
  
  function legion_point_3d_constructor_integer8(x, y, z)
    implicit none
    
    type(FPoint3D)              :: legion_point_3d_constructor_integer8
    integer(kind=8), intent(in) :: x
    integer(kind=8), intent(in) :: y
    integer(kind=8), intent(in) :: z
    
    legion_point_3d_constructor_integer8%point%x(0) = x
    legion_point_3d_constructor_integer8%point%x(1) = y
    legion_point_3d_constructor_integer8%point%x(2) = z
      
  end function legion_point_3d_constructor_integer8

  function legion_field_accessor_1d_constructor(physical_region, fid, data_size)
    implicit none
    
    type(FFieldAccessor1D)            :: legion_field_accessor_1d_constructor
    type(FPhysicalRegion), intent(in) :: physical_region
    integer(c_int), intent(in)        :: fid
    integer(c_size_t), intent(in)     :: data_size
    

    legion_field_accessor_1d_constructor%dim = 1
    legion_field_accessor_1d_constructor%data_size = data_size
    legion_field_accessor_1d_constructor%accessor = legion_physical_region_get_field_accessor_array_1d_c(physical_region%pr, fid)
  end function legion_field_accessor_1d_constructor
  
  function legion_field_accessor_2d_constructor(physical_region, fid, data_size)
    implicit none
    
    type(FFieldAccessor2D)            :: legion_field_accessor_2d_constructor
    type(FPhysicalRegion), intent(in) :: physical_region
    integer(c_int), intent(in)        :: fid
    integer(c_size_t), intent(in)     :: data_size
    

    legion_field_accessor_2d_constructor%dim = 1
    legion_field_accessor_2d_constructor%data_size = data_size
    legion_field_accessor_2d_constructor%accessor = legion_physical_region_get_field_accessor_array_2d_c(physical_region%pr, fid)
  end function legion_field_accessor_2d_constructor
  
  function legion_field_accessor_3d_constructor(physical_region, fid, data_size)
    implicit none
    
    type(FFieldAccessor3D)            :: legion_field_accessor_3d_constructor
    type(FPhysicalRegion), intent(in) :: physical_region
    integer(c_int), intent(in)        :: fid
    integer(c_size_t), intent(in)     :: data_size
    

    legion_field_accessor_3d_constructor%dim = 1
    legion_field_accessor_3d_constructor%data_size = data_size
    legion_field_accessor_3d_constructor%accessor = legion_physical_region_get_field_accessor_array_3d_c(physical_region%pr, fid)
  end function legion_field_accessor_3d_constructor
      
  subroutine legion_field_accessor_init(this, physical_region, fid, data_size)
    implicit none
    
    class(FFieldAccessor), intent(inout)    :: this
    type(legion_physical_region_f_t), intent(in) :: physical_region
    integer(c_int), intent(in)                   :: fid
    integer(c_size_t), intent(in)                :: data_size
    
    select type (this)
    type is (FFieldAccessor1D)
      ! 1D
      this%dim = 1
      this%data_size = data_size
      this%accessor = legion_physical_region_get_field_accessor_array_1d_c(physical_region, fid)
    type is (FFieldAccessor2D)
      ! 2D
      this%dim = 2
      this%data_size = data_size
      this%accessor = legion_physical_region_get_field_accessor_array_2d_c(physical_region, fid)
    type is (FFieldAccessor3D)
      ! 3D
      this%dim = 3
      this%data_size = data_size
      this%accessor = legion_physical_region_get_field_accessor_array_3d_c(physical_region, fid)
    class default
      ! give error for unexpected/unsupported type
      stop 'initialize: unexpected type for LegionFieldAccessor object!'
    end select
  end subroutine legion_field_accessor_init
  
  subroutine legion_field_accessor_read_point_ptr(this, point, dst)
    implicit none
    
    class(FFieldAccessor), intent(in) :: this
    class(FPoint), intent(in)         :: point
    type(c_ptr)                            :: dst 
    
    select type (this)
    type is (FFieldAccessor1D)
      ! 1D
      select type (point)
      type is (FPoint1D)
        call legion_accessor_array_1d_read_point_c(this%accessor, point%point, dst, this%data_size)
      end select
    type is (FFieldAccessor2D)
      ! 2D
      select type (point)
      type is (FPoint2D)
        call legion_accessor_array_2d_read_point_c(this%accessor, point%point, dst, this%data_size)
      end select
    type is (FFieldAccessor3D)
      ! 3D
      select type (point)
      type is (FPoint3D)
        call legion_accessor_array_3d_read_point_c(this%accessor, point%point, dst, this%data_size)
      end select
    class default
      ! give error for unexpected/unsupported type
      stop 'initialize: unexpected type for LegionFieldAccessor object!'
    end select
  end subroutine legion_field_accessor_read_point_ptr
  
  subroutine legion_field_accessor_read_point_integer4(this, point, dst)
    implicit none
    
    class(FFieldAccessor), intent(in)    :: this
    class(FPoint), intent(in)            :: point
    integer(kind=4), target, intent(out) :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_integer4
  
  subroutine legion_field_accessor_read_point_integer8(this, point, dst)
    implicit none
    
    class(FFieldAccessor), intent(in)    :: this
    class(FPoint), intent(in)            :: point
    integer(kind=8), target, intent(out) :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_integer8
  
  subroutine legion_field_accessor_read_point_real4(this, point, dst)
    implicit none
    
    class(FFieldAccessor), intent(in) :: this
    class(FPoint), intent(in)         :: point
    real(kind=4), target, intent(out) :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_real4
  
  subroutine legion_field_accessor_read_point_real8(this, point, dst)
    implicit none
    
    class(FFieldAccessor), intent(in) :: this
    class(FPoint), intent(in)         :: point
    real(kind=8), target, intent(out) :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_real8
  
  subroutine legion_field_accessor_read_point_complex4(this, point, dst)
    implicit none
    
    class(FFieldAccessor), intent(in)    :: this
    class(FPoint), intent(in)            :: point
    complex(kind=4), target, intent(out) :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_complex4
  
  subroutine legion_field_accessor_read_point_complex8(this, point, dst)
    implicit none
    
    class(FFieldAccessor), intent(in)    :: this
    class(FPoint), intent(in)            :: point
    complex(kind=8), target, intent(out) :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_complex8
    
  subroutine legion_field_accessor_write_point_ptr(this, point, src)
    implicit none

    class(FFieldAccessor), intent(in) :: this
    class(FPoint), intent(in)         :: point
    type(c_ptr), intent(in)           :: src
    
    select type (this)
    type is (FFieldAccessor1D)
      ! 1D
      select type (point)
      type is (FPoint1D)
        call legion_accessor_array_1d_write_point_c(this%accessor, point%point, src, this%data_size)
      end select
    type is (FFieldAccessor2D)
      ! 2D
      select type (point)
      type is (FPoint2D)
        call legion_accessor_array_2d_write_point_c(this%accessor, point%point, src, this%data_size)
      end select
    type is (FFieldAccessor3D)
      ! 3D
      select type (point)
      type is (FPoint3D)
        call legion_accessor_array_3d_write_point_c(this%accessor, point%point, src, this%data_size)
      end select
    class default
      ! give error for unexpected/unsupported type
         stop 'initialize: unexpected type for LegionFieldAccessor object!'
    end select
  end subroutine legion_field_accessor_write_point_ptr
  
  subroutine legion_field_accessor_write_point_integer4(this, point, src)
    implicit none

    class(FFieldAccessor), intent(in)   :: this
    class(FPoint), intent(in)           :: point
    integer(kind=4), target, intent(in) :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_integer4
  
  subroutine legion_field_accessor_write_point_integer8(this, point, src)
    implicit none

    class(FFieldAccessor), intent(in)   :: this
    class(FPoint), intent(in)           :: point
    integer(kind=8), target, intent(in) :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_integer8
  
  subroutine legion_field_accessor_write_point_real4(this, point, src)
    implicit none

    class(FFieldAccessor), intent(in) :: this
    class(FPoint), intent(in)         :: point
    real(kind=4), target, intent(in)  :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_real4
  
  subroutine legion_field_accessor_write_point_real8(this, point, src)
    implicit none

    class(FFieldAccessor), intent(in) :: this
    class(FPoint), intent(in)         :: point
    real(kind=8), target, intent(in)  :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_real8
  
  subroutine legion_field_accessor_write_point_complex4(this, point, src)
    implicit none

    class(FFieldAccessor), intent(in)   :: this
    class(FPoint), intent(in)           :: point
    complex(kind=4), target, intent(in) :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_complex4
  
  subroutine legion_field_accessor_write_point_complex8(this, point, src)
    implicit none

    class(FFieldAccessor), intent(in)   :: this
    class(FPoint), intent(in)           :: point
    complex(kind=8), target, intent(in) :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_complex8
  
  function legion_array_2d_real8_constructor(raw_ptr, num_rows, num_columns, ld)
    implicit none
    
    type(LegionArray2DReal8)                 :: legion_array_2d_real8_constructor
    type(c_ptr), intent(in)                  :: raw_ptr
    integer, intent(in)                      :: num_rows
    integer, intent(in)                      :: num_columns
    type(legion_byte_offset_f_t), intent(in) :: ld
        
    type(c_ptr), allocatable :: ptr_columns(:)
    type(LegionArray2DReal8) :: tmp_2d_array
    real(kind=8), pointer :: column(:)
    real(kind=8), target :: col(10)
    integer :: i
    type(CellReal8), allocatable :: matrix(:)
    
    allocate(matrix(num_columns))
    
 !   allocate(tmp_2d_array%x(1:num_columns))
    col(1:10) = 1
    tmp_2d_array%x(1)%y(1:10) => col
    matrix(1)%y =>col
    column => col
        
    tmp_2d_array%dim_x = num_columns
    tmp_2d_array%dim_y = num_rows
    tmp_2d_array%ld = ld%offset
    allocate(ptr_columns(num_columns))
    call legion_convert_1d_to_2d_column_major_c(raw_ptr, ptr_columns, ld, num_columns)
    do i = 1, num_columns
      call c_f_pointer(ptr_columns(i), column, [num_rows])
      call c_f_pointer(ptr_columns(i), tmp_2d_array%x(i)%y, [num_rows])
    end do
    legion_array_2d_real8_constructor = tmp_2d_array
  end function legion_array_2d_real8_constructor
  
  ! ======== LogicalRegion ========
  function legion_logical_region_get_index_space(this)
    implicit none
    
    type(FIndexSpace)                 :: legion_logical_region_get_index_space 
    class(FLogicalRegion), intent(in) :: this
    
    legion_logical_region_get_index_space%is = legion_logical_region_get_index_space_c(this%lr)
  end function legion_logical_region_get_index_space
  
  ! ======== PhysicalRegionList ========
  function legion_physical_region_list_size(this)
    implicit none
    
    type(integer)                         :: legion_physical_region_list_size
    class(FPhysicalRegionList), intent(in) :: this
      
    legion_physical_region_list_size = this%num_regions
  end function legion_physical_region_list_size
  
  function legion_physical_region_list_get_region_by_id(this, id)
    implicit none
    
    type(FPhysicalRegion)                  :: legion_physical_region_list_get_region_by_id
    class(FPhysicalRegionList), intent(in) :: this
    type(integer), intent(in)             :: id
      
    legion_physical_region_list_get_region_by_id%pr = legion_get_physical_region_by_id_c(this%region_ptr, id, this%num_regions)
  end function legion_physical_region_list_get_region_by_id
  
  ! ======== RegionRequirement ========
  function legion_region_requirement_get_logical_region(this)
    implicit none
    
    type(FLogicalRegion)                  :: legion_region_requirement_get_logical_region
    class(FRegionRequirement), intent(in) :: this
      
    legion_region_requirement_get_logical_region%lr = legion_region_requirement_get_region_c(this%rr)
  end function legion_region_requirement_get_logical_region
  
  ! ======== LegionTask ========
  function legion_task_get_region_requirement_by_id(this, id)
    implicit none
    
    type(FRegionRequirement)       :: legion_task_get_region_requirement_by_id
    class(FTask), intent(in) :: this
    type(integer), intent(in)     :: id
      
    legion_task_get_region_requirement_by_id%rr = legion_task_get_requirement_c(this%task, id)
    
  end function legion_task_get_region_requirement_by_id
  
  ! ======== LegionRuntime ========
  function legion_runtime_get_index_domain_return_domain(this, ctx, index_space)
    implicit none
    
    type(FDomain)                :: legion_runtime_get_index_domain_return_domain
    class(FRuntime), intent(in)   :: this
    type(FContext), intent(in)    :: ctx
    type(FIndexSpace), intent(in) :: index_space
    
    legion_runtime_get_index_domain_return_domain%domain = legion_index_space_get_domain_c(this%runtime, index_space%is)
  end function legion_runtime_get_index_domain_return_domain
  
  function legion_runtime_get_index_domain_return_rect1(this, ctx, index_space, dim)
    implicit none
    
    type(FRect1D)                 :: legion_runtime_get_index_domain_return_rect1
    class(FRuntime), intent(in)   :: this
    type(FContext), intent(in)    :: ctx
    type(FIndexSpace), intent(in) :: index_space
    type(integer), intent(in)     :: dim
    type(legion_domain_f_t) :: tmp_domain
    
    tmp_domain = legion_index_space_get_domain_c(this%runtime, index_space%is)
    legion_runtime_get_index_domain_return_rect1%domain = tmp_domain
    legion_runtime_get_index_domain_return_rect1%rect = legion_domain_get_rect_1d_c(tmp_domain)
  end function legion_runtime_get_index_domain_return_rect1
  
  ! ================================
  subroutine legion_task_prolog(tdata, tdatalen, proc_id, &
                                task, pr_list, &
                                ctx, runtime)
    
    implicit none
                                
    type(c_ptr), intent(in)                         :: tdata ! pass reference
    integer(c_size_t), value, intent(in)            :: tdatalen
    integer(c_long_long), value, intent(in)         :: proc_id
    type(FTask), intent(out)                   :: task ! pass reference
    type(FPhysicalRegionList), intent(out)           :: pr_list ! pass reference
    type(FContext), intent(out)                      :: ctx ! pass reference          
    type(FRuntime), intent(out)                      :: runtime ! pass reference
      
    call legion_task_preamble_c(tdata, tdatalen, proc_id, &
                                task%task, pr_list%region_ptr, pr_list%num_regions, &
                                ctx%context, runtime%runtime)
  end subroutine legion_task_prolog
  
  subroutine legion_task_epilog(runtime, ctx, retval, retsize)
    implicit none

    type(FRuntime), intent(in) :: runtime
    type(FContext), intent(in) :: ctx
    type(c_ptr), value, intent(in)              :: retval
    integer(c_size_t), value, intent(in)        :: retsize
  
    call legion_task_postamble_c(runtime%runtime, ctx%context, retval, retsize)
  end subroutine legion_task_epilog

end module