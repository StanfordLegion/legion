module legion_fortran_object_oriented
  use, intrinsic :: iso_c_binding
  use legion_fortran_types
  use legion_fortran_c_interface
  implicit none
  
  ! Point Class
  type LegionPoint
    integer :: dim
  end type LegionPoint

  type, extends(LegionPoint) :: LegionPoint1D
    type(legion_point_1d_f_t) :: point
  end type LegionPoint1D
  
  type, extends(LegionPoint) :: LegionPoint2D
    type(legion_point_2d_f_t) :: point
  end type LegionPoint2D
  
  type, extends(LegionPoint) :: LegionPoint3D
    type(legion_point_3d_f_t) :: point
  end type LegionPoint3D
  
  interface LegionPoint1D
    module procedure legion_point_1d_constructor_integer4
    module procedure legion_point_1d_constructor_integer8
  end interface
  
  interface LegionPoint2D
    module procedure legion_point_2d_constructor_integer4
    module procedure legion_point_2d_constructor_integer8
  end interface
  
  interface LegionPoint3D
    module procedure legion_point_3d_constructor_integer4
    module procedure legion_point_3d_constructor_integer8
  end interface
  
  ! Accessor Class
  type LegionFieldAccessor
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
  end type LegionFieldAccessor

  type, extends(LegionFieldAccessor) :: LegionFieldAccessor1D
    type(legion_accessor_array_1d_f_t) :: accessor
  end type LegionFieldAccessor1D
  
  type, extends(LegionFieldAccessor) :: LegionFieldAccessor2D
    type(legion_accessor_array_2d_f_t) :: accessor
  end type LegionFieldAccessor2D
  
  type, extends(LegionFieldAccessor) :: LegionFieldAccessor3D
    type(legion_accessor_array_3d_f_t) :: accessor
  end type LegionFieldAccessor3D
  
  interface LegionFieldAccessor1D
    module procedure legion_field_accessor_1d_constructor
  end interface
  
  interface LegionFieldAccessor2D
    module procedure legion_field_accessor_2d_constructor
  end interface
  
  interface LegionFieldAccessor3D
    module procedure legion_field_accessor_3d_constructor
  end interface
  
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
    
    type(LegionPoint1D)         :: legion_point_1d_constructor_integer4
    integer(kind=4), intent(in) :: x
    
    legion_point_1d_constructor_integer4%point%x(0) = int(x, 8)
      
  end function legion_point_1d_constructor_integer4
  
  function legion_point_1d_constructor_integer8(x)
    implicit none
    
    type(LegionPoint1D)         :: legion_point_1d_constructor_integer8
    integer(kind=8), intent(in) :: x
    
    legion_point_1d_constructor_integer8%point%x(0) = x
      
  end function legion_point_1d_constructor_integer8
  
  function legion_point_2d_constructor_integer4(x, y)
    implicit none
    
    type(LegionPoint2D)         :: legion_point_2d_constructor_integer4
    integer(kind=4), intent(in) :: x
    integer(kind=4), intent(in) :: y
    
    legion_point_2d_constructor_integer4%point%x(0) = int(x, 8)
    legion_point_2d_constructor_integer4%point%x(1) = int(y, 8)
      
  end function legion_point_2d_constructor_integer4
  
  function legion_point_2d_constructor_integer8(x, y)
    implicit none
    
    type(LegionPoint2D)         :: legion_point_2d_constructor_integer8
    integer(kind=8), intent(in) :: x
    integer(kind=8), intent(in) :: y
    
    legion_point_2d_constructor_integer8%point%x(0) = x
    legion_point_2d_constructor_integer8%point%x(1) = y
      
  end function legion_point_2d_constructor_integer8
  
  function legion_point_3d_constructor_integer4(x, y, z)
    implicit none
    
    type(LegionPoint3D)         :: legion_point_3d_constructor_integer4
    integer(kind=4), intent(in) :: x
    integer(kind=4), intent(in) :: y
    integer(kind=4), intent(in) :: z
    
    legion_point_3d_constructor_integer4%point%x(0) = int(x, 8)
    legion_point_3d_constructor_integer4%point%x(1) = int(y, 8)
    legion_point_3d_constructor_integer4%point%x(2) = int(z, 8)
      
  end function legion_point_3d_constructor_integer4
  
  function legion_point_3d_constructor_integer8(x, y, z)
    implicit none
    
    type(LegionPoint3D)         :: legion_point_3d_constructor_integer8
    integer(kind=8), intent(in) :: x
    integer(kind=8), intent(in) :: y
    integer(kind=8), intent(in) :: z
    
    legion_point_3d_constructor_integer8%point%x(0) = x
    legion_point_3d_constructor_integer8%point%x(1) = y
    legion_point_3d_constructor_integer8%point%x(2) = z
      
  end function legion_point_3d_constructor_integer8

  function legion_field_accessor_1d_constructor(physical_region, fid, data_size)
    implicit none
    
    type(LegionFieldAccessor1D)                  :: legion_field_accessor_1d_constructor
    type(legion_physical_region_f_t), intent(in) :: physical_region
    integer(c_int), intent(in)                   :: fid
    integer(c_size_t), intent(in)                :: data_size
    

    legion_field_accessor_1d_constructor%dim = 1
    legion_field_accessor_1d_constructor%data_size = data_size
    legion_field_accessor_1d_constructor%accessor = legion_physical_region_get_field_accessor_array_1d_c(physical_region, fid)
  end function legion_field_accessor_1d_constructor
  
  function legion_field_accessor_2d_constructor(physical_region, fid, data_size)
    implicit none
    
    type(LegionFieldAccessor2D)                  :: legion_field_accessor_2d_constructor
    type(legion_physical_region_f_t), intent(in) :: physical_region
    integer(c_int), intent(in)                   :: fid
    integer(c_size_t), intent(in)                :: data_size
    

    legion_field_accessor_2d_constructor%dim = 1
    legion_field_accessor_2d_constructor%data_size = data_size
    legion_field_accessor_2d_constructor%accessor = legion_physical_region_get_field_accessor_array_2d_c(physical_region, fid)
  end function legion_field_accessor_2d_constructor
  
  function legion_field_accessor_3d_constructor(physical_region, fid, data_size)
    implicit none
    
    type(LegionFieldAccessor3D)                  :: legion_field_accessor_3d_constructor
    type(legion_physical_region_f_t), intent(in) :: physical_region
    integer(c_int), intent(in)                   :: fid
    integer(c_size_t), intent(in)                :: data_size
    

    legion_field_accessor_3d_constructor%dim = 1
    legion_field_accessor_3d_constructor%data_size = data_size
    legion_field_accessor_3d_constructor%accessor = legion_physical_region_get_field_accessor_array_3d_c(physical_region, fid)
  end function legion_field_accessor_3d_constructor
      
  subroutine legion_field_accessor_init(this, physical_region, fid, data_size)
    implicit none
    
    class(LegionFieldAccessor), intent(inout)    :: this
    type(legion_physical_region_f_t), intent(in) :: physical_region
    integer(c_int), intent(in)                   :: fid
    integer(c_size_t), intent(in)                :: data_size
    
    select type (this)
    type is (LegionFieldAccessor1D)
      ! 1D
      this%dim = 1
      this%data_size = data_size
      this%accessor = legion_physical_region_get_field_accessor_array_1d_c(physical_region, fid)
    type is (LegionFieldAccessor2D)
      ! 2D
      this%dim = 2
      this%data_size = data_size
      this%accessor = legion_physical_region_get_field_accessor_array_2d_c(physical_region, fid)
    type is (LegionFieldAccessor3D)
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
    
    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    type(c_ptr)                            :: dst 
    
    select type (this)
    type is (LegionFieldAccessor1D)
      ! 1D
      select type (point)
      type is (LegionPoint1D)
        call legion_accessor_array_1d_read_point_c(this%accessor, point%point, dst, this%data_size)
      end select
    type is (LegionFieldAccessor2D)
      ! 2D
      select type (point)
      type is (LegionPoint2D)
        call legion_accessor_array_2d_read_point_c(this%accessor, point%point, dst, this%data_size)
      end select
    type is (LegionFieldAccessor3D)
      ! 3D
      select type (point)
      type is (LegionPoint3D)
        call legion_accessor_array_3d_read_point_c(this%accessor, point%point, dst, this%data_size)
      end select
    class default
      ! give error for unexpected/unsupported type
      stop 'initialize: unexpected type for LegionFieldAccessor object!'
    end select
  end subroutine legion_field_accessor_read_point_ptr
  
  subroutine legion_field_accessor_read_point_integer4(this, point, dst)
    implicit none
    
    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    integer(kind=4), target, intent(out)   :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_integer4
  
  subroutine legion_field_accessor_read_point_integer8(this, point, dst)
    implicit none
    
    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    integer(kind=8), target, intent(out)   :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_integer8
  
  subroutine legion_field_accessor_read_point_real4(this, point, dst)
    implicit none
    
    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    real(kind=4), target, intent(out)      :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_real4
  
  subroutine legion_field_accessor_read_point_real8(this, point, dst)
    implicit none
    
    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    real(kind=8), target, intent(out)      :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_real8
  
  subroutine legion_field_accessor_read_point_complex4(this, point, dst)
    implicit none
    
    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    complex(kind=4), target, intent(out)      :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_complex4
  
  subroutine legion_field_accessor_read_point_complex8(this, point, dst)
    implicit none
    
    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    complex(kind=8), target, intent(out)      :: dst
    
    call legion_field_accessor_read_point_ptr(this, point, c_loc(dst))
  end subroutine legion_field_accessor_read_point_complex8
    
  subroutine legion_field_accessor_write_point_ptr(this, point, src)
    implicit none

    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    type(c_ptr), intent(in)                :: src
    
    select type (this)
    type is (LegionFieldAccessor1D)
      ! 1D
      select type (point)
      type is (LegionPoint1D)
        call legion_accessor_array_1d_write_point_c(this%accessor, point%point, src, this%data_size)
      end select
    type is (LegionFieldAccessor2D)
      ! 2D
      select type (point)
      type is (LegionPoint2D)
        call legion_accessor_array_2d_write_point_c(this%accessor, point%point, src, this%data_size)
      end select
    type is (LegionFieldAccessor3D)
      ! 3D
      select type (point)
      type is (LegionPoint3D)
        call legion_accessor_array_3d_write_point_c(this%accessor, point%point, src, this%data_size)
      end select
    class default
      ! give error for unexpected/unsupported type
         stop 'initialize: unexpected type for LegionFieldAccessor object!'
    end select
  end subroutine legion_field_accessor_write_point_ptr
  
  subroutine legion_field_accessor_write_point_integer4(this, point, src)
    implicit none

    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    integer(kind=4), target, intent(in)    :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_integer4
  
  subroutine legion_field_accessor_write_point_integer8(this, point, src)
    implicit none

    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    integer(kind=8), target, intent(in)    :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_integer8
  
  subroutine legion_field_accessor_write_point_real4(this, point, src)
    implicit none

    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    real(kind=4), target, intent(in)       :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_real4
  
  subroutine legion_field_accessor_write_point_real8(this, point, src)
    implicit none

    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    real(kind=8), target, intent(in)       :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_real8
  
  subroutine legion_field_accessor_write_point_complex4(this, point, src)
    implicit none

    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    complex(kind=4), target, intent(in)       :: src
    
    call legion_field_accessor_write_point_ptr(this, point, c_loc(src))
  end subroutine legion_field_accessor_write_point_complex4
  
  subroutine legion_field_accessor_write_point_complex8(this, point, src)
    implicit none

    class(LegionFieldAccessor), intent(in) :: this
    class(LegionPoint), intent(in)         :: point
    complex(kind=8), target, intent(in)       :: src
    
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

end module