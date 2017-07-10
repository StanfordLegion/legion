subroutine daxpy_task_f(alpha, n, x_ptr, y_ptr, z_ptr) bind(C)
    use iso_c_binding, only: c_double, c_int

    real(c_double), intent(in)                  :: alpha
    integer(kind=c_int), intent(in)             :: n
    real(c_double), intent(in), dimension(n)    :: x_ptr
    real(c_double), intent(in), dimension(n)    :: y_ptr
    real(c_double), intent(out), dimension(n)    :: z_ptr

    integer         :: i
    real(c_double)  ::  value
    do i = 1, n
        value = alpha * x_ptr(i) + y_ptr(i)
        z_ptr(i) = value
    end do

end subroutine daxpy_task_f
