subroutine hello_world_task_f(rank, n, cstring, rt_val) bind(C)
    use iso_c_binding, only: c_ptr, c_int, c_f_pointer, c_loc, c_null_char
    implicit none

    integer(kind=c_int),               intent(in) :: n
    integer(kind=c_int),               intent(in) :: rank
    type(c_ptr), dimension(n), target, intent(in) :: cstring
    integer(kind=c_int),               intent(out) :: rt_val
    character, pointer                            :: fstring(:)
    integer                                       :: i, j
    character, allocatable                        :: print_string(:)

    allocate(print_string(n*5))
    do i = 1, n
      call c_f_pointer(cstring(i), fstring, [5])
      do j = 1, 5
          print_string((i-1)*5+j:(i-1)*5+j) = fstring(j)
      end do
    end do
    
    print *, print_string, rank
    
    rt_val = rank;

end subroutine hello_world_task_f
