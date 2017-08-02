module legion_fortran
    use, intrinsic :: iso_c_binding
    use legion_fortran_types
    use legion_fortran_c_interface
    implicit none
    
    interface legion_accessor_array_1d_read_point_f
        module procedure legion_accessor_array_1d_read_point_ptr_f
        module procedure legion_accessor_array_1d_read_point_integer2_f
        module procedure legion_accessor_array_1d_read_point_integer4_f
        module procedure legion_accessor_array_1d_read_point_integer8_f
        module procedure legion_accessor_array_1d_read_point_real4_f
        module procedure legion_accessor_array_1d_read_point_real8_f
        module procedure legion_accessor_array_1d_read_point_complex4_f
        module procedure legion_accessor_array_1d_read_point_complex8_f
    end interface
    
    type LegionPoint
        integer :: dim
    end type LegionPoint

    type, extends(LegionPoint) :: LegionPoint1D
        type(legion_point_1d_f_t) :: point
    end type LegionPoint1D
    
    type LegionFieldAccessor
        integer :: dim
        integer(c_size_t) :: data_size
    contains
        procedure :: init => legion_field_accessor_init
        procedure, private :: legion_field_accessor_read_point_ptr
        procedure, private :: legion_field_accessor_read_point_real8
        generic :: read_point => legion_field_accessor_read_point_ptr, legion_field_accessor_read_point_real8
    end type LegionFieldAccessor

    type, extends(LegionFieldAccessor) :: LegionFieldAccessor1D
        type(legion_accessor_array_1d_f_t) :: accessor
    end type LegionFieldAccessor1D
    
contains
    
    subroutine legion_field_accessor_init(this, physical_region, fid, data_size)
        implicit none
        
        class(LegionFieldAccessor), intent(inout)    :: this
        type(legion_physical_region_f_t), intent(in) :: physical_region
        integer(c_int), intent(in)                   :: fid
        integer(c_size_t), intent(in)                  :: data_size
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            this%dim = 1
            this%data_size = data_size
            this%accessor = legion_physical_region_get_field_accessor_array_1d_c(physical_region, fid)
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_init
    
    subroutine legion_field_accessor_read_point_ptr(this, point, dst)
        implicit none
        
        class(LegionFieldAccessor), intent(in)    :: this
        class(LegionPoint), intent(in)            :: point
        type(c_ptr), intent(out)                  :: dst 
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_read_point_c(this%accessor, point%point, dst, this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_read_point_ptr
    
    subroutine legion_field_accessor_read_point_real8(this, point, dst)
        implicit none
        
        class(LegionFieldAccessor), intent(in)    :: this
        class(LegionPoint), intent(in)            :: point
        real(kind=8), target, intent(out)         :: dst
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_read_point_c(this%accessor, point%point, c_loc(dst), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_read_point_real8
    
    ! -----------------------------------------------------------------------
    ! Task Launcher
    ! -----------------------------------------------------------------------
    ! @see Legion::TaskLauncher::TaskLauncher()
    subroutine legion_task_launcher_create_f(tid, arg, pred, id, tag, task_launcher)
        implicit none
        
        integer(c_int), intent(in)                  :: tid
        type(legion_task_argument_f_t), intent(in)  :: arg
        type(legion_predicate_f_t), intent(in)      :: pred
        integer(c_int), intent(in)                  :: id
        integer(c_long), intent(in)                 :: tag
        type(legion_task_launcher_f_t), intent(out) :: task_launcher

        task_launcher = legion_task_launcher_create_c(tid, arg, pred, id, tag)
    end subroutine legion_task_launcher_create_f
    
    ! @see Legion::TaskLauncher::~TaskLauncher()
    subroutine legion_task_launcher_destroy_f(handle)
        implicit none
        
        type(legion_task_launcher_f_t), value, intent(in) :: handle

        call legion_task_launcher_destroy_c(handle)
    end subroutine legion_task_launcher_destroy_f
    
    ! @see Legion::Runtime::execute_task()
    subroutine legion_task_launcher_execute_f(runtime, ctx, launcher, future)
        implicit none
        
        type(legion_runtime_f_t), intent(in)       :: runtime
        type(legion_context_f_t), intent(in)       :: ctx
        type(legion_task_launcher_f_t), intent(in) :: launcher
        type(legion_future_f_t), intent(out)       :: future
        
        future = legion_task_launcher_execute_c(runtime, ctx, launcher)
    end subroutine legion_task_launcher_execute_f
    
    ! @see Legion::TaskLauncher::add_region_requirement()
    subroutine legion_task_launcher_add_region_requirement_logical_region_f(launcher, handle, priv, prop, &
            parent, tag, verified, rr_idx)
        implicit none
        
        type(legion_task_launcher_f_t), intent(in)  :: launcher
        type(legion_logical_region_f_t), intent(in) :: handle
        integer(c_int), intent(in)                  :: priv
        integer(c_int), intent(in)                  :: prop
        type(legion_logical_region_f_t), intent(in) :: parent
        integer(c_long), intent(in)                 :: tag
        logical(c_bool), intent(in)                 :: verified
        integer(c_int), intent(out)                 :: rr_idx
      
        rr_idx = legion_task_launcher_add_region_requirement_logical_region_c(launcher, handle, priv, prop, parent, tag, verified)
    end subroutine legion_task_launcher_add_region_requirement_logical_region_f
    
    ! @see Legion::TaskLaunchxer::add_field()
    subroutine legion_task_launcher_add_field_f(launcher, idx, fid, inst)
        implicit none
        
        type(legion_task_launcher_f_t), intent(in) :: launcher
        integer(c_int), intent(in)                 :: idx
        integer(c_int), intent(in)                 :: fid
        logical(c_bool), intent(in)                :: inst

        call legion_task_launcher_add_field_c(launcher, idx, fid, inst)
    end subroutine legion_task_launcher_add_field_f
    
    ! -----------------------------------------------------------------------
    ! Index Launcher
    ! -----------------------------------------------------------------------
    ! @see Legion::IndexTaskLauncher::IndexTaskLauncher()
    subroutine legion_index_launcher_create_f(tid, domain, global_arg, map, pred, must, id, tag, index_launcher)
        implicit none
    
        integer(c_int), intent(in)                   :: tid
        type(legion_domain_f_t), intent(in)          :: domain
        type(legion_task_argument_f_t), intent(in)   :: global_arg
        type(legion_argument_map_f_t), intent(in)    :: map
        type(legion_predicate_f_t), intent(in)       :: pred
        logical(c_bool), intent(in)                  :: must
        integer(c_int), intent(in)                   :: id
        integer(c_long), intent(in)                  :: tag
        type(legion_index_launcher_f_t), intent(out) :: index_launcher
            
        index_launcher = legion_index_launcher_create_c(tid, domain, global_arg, map, pred, must, id, tag)
    end subroutine legion_index_launcher_create_f
    
    ! @see Legion::IndexTaskLauncher::~IndexTaskLauncher()
    subroutine legion_index_launcher_destroy_f(handle)
        implicit none
    
        type(legion_index_launcher_f_t), intent(in) :: handle
            
        call legion_index_launcher_destroy_c(handle)
    end subroutine legion_index_launcher_destroy_f
    
    ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &)
    subroutine legion_index_launcher_execute_f(runtime, ctx, launcher, future_map)
        implicit none
    
        type(legion_runtime_f_t), intent(in)         :: runtime
        type(legion_context_f_t), intent(in)         :: ctx
        type(legion_index_launcher_f_t), intent(in)  :: launcher
        type(legion_future_map_f_t), intent(out)     :: future_map
            
        future_map = legion_index_launcher_execute_c(runtime, ctx, launcher)
    end subroutine legion_index_launcher_execute_f
    
    ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, ReductionOpID)
    subroutine legion_index_launcher_execute_reduction_f(runtime, ctx, launcher, redop, future)
        implicit none
    
        type(legion_runtime_f_t), intent(in)        :: runtime
        type(legion_context_f_t), intent(in)        :: ctx
        type(legion_index_launcher_f_t), intent(in) :: launcher
        integer(c_int), intent(in)                  :: redop 
        type(legion_future_f_t), intent(out)        :: future
            
        future = legion_index_launcher_execute_reduction_c(runtime, ctx, launcher, redop)
    end subroutine legion_index_launcher_execute_reduction_f
    
    ! @see Legion::IndexTaskLauncher::add_region_requirement()
    subroutine legion_index_launcher_add_region_requirement_lp_f(launcher, handle, proj, priv, &
            prop, parent, tag, verified, rr_idx)
        implicit none
    
        type(legion_index_launcher_f_t), intent(in)    :: launcher
        type(legion_logical_partition_f_t), intent(in) :: handle
        integer(c_int), intent(in)                     :: proj
        integer(c_int), intent(in)                     :: priv
        integer(c_int), intent(in)                     :: prop
        type(legion_logical_region_f_t), intent(in)    :: parent
        integer(c_long), intent(in)                    :: tag
        logical(c_bool), intent(in)                    :: verified
        integer(c_int), intent(out)                    :: rr_idx
        
        rr_idx = legion_index_launcher_add_region_requirement_lp_c(launcher, handle, proj, priv, &
            prop, parent, tag, verified)
    end subroutine legion_index_launcher_add_region_requirement_lp_f
    
    ! @see Legion::TaskLaunchxer::add_field()
    subroutine legion_index_launcher_add_field_f(launcher, idx, fid, inst) 
        implicit none
    
        type(legion_index_launcher_f_t), intent(in) :: launcher
        integer(c_int), intent(in)                  :: idx
        integer(c_int), intent(in)                  :: fid
        logical(c_bool), intent(in)                 :: inst 
    
        call legion_index_launcher_add_field_c(launcher, idx, fid, inst) 
    end subroutine legion_index_launcher_add_field_f
    
    ! -----------------------------------------------------------------------
    ! Predicate Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Predicate::TRUE_PRED
    subroutine legion_predicate_true_f(pred)
        implicit none
    
        type(legion_predicate_f_t), intent(out)  :: pred
            
        pred = legion_predicate_true_c()
    end subroutine legion_predicate_true_f

    ! @see Legion::Predicate::FALSE_PRED
    subroutine legion_predicate_false_f(pred)
        implicit none
    
        type(legion_predicate_f_t), intent(out)  :: pred
        pred = legion_predicate_false_c()
    end subroutine legion_predicate_false_f
    
    ! -----------------------------------------------------------------------
    ! Argument Map
    ! -----------------------------------------------------------------------
    ! @see Legion::ArgumentMap::ArgumentMap()
    subroutine legion_argument_map_create_f(arg_map)
        implicit none
    
        type(legion_argument_map_f_t), intent(out) :: arg_map
        
        arg_map = legion_argument_map_create_c()
    end subroutine legion_argument_map_create_f

    ! @see Legion::ArgumentMap::set_point()
    subroutine legion_argument_map_set_point_f(map, dp, arg, replace)
        implicit none
    
        type(legion_argument_map_f_t), intent(in)  :: map
        type(legion_domain_point_f_t), intent(in)  :: dp 
        type(legion_task_argument_f_t), intent(in) :: arg
        logical(c_bool), intent(in)                :: replace 
        
        call legion_argument_map_set_point_c(map, dp, arg, replace)
    end subroutine legion_argument_map_set_point_f
    
    ! -----------------------------------------------------------------------
    ! Task Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Task::args
    subroutine legion_task_get_args_f(task, args)
        implicit none
    
        type(legion_task_f_t), intent(in) :: task
        type(c_ptr), intent(out)          :: args
            
        args = legion_task_get_args_c(task)
    end subroutine legion_task_get_args_f

    ! @see Legion::Task::arglen
    subroutine legion_task_get_arglen_f(task, arglen)
        implicit none
    
        type(legion_task_f_t), intent(in) :: task
        integer(c_size_t), intent(out)    :: arglen
        
        arglen = legion_task_get_arglen_c(task)
    end subroutine legion_task_get_arglen_f

    ! @see Legion::Task::local_args
    subroutine legion_task_get_local_args_f(task, local_args)
        implicit none
    
        type(legion_task_f_t), intent(in) :: task
        type(c_ptr), intent(out)          :: local_args
            
        local_args = legion_task_get_local_args_c(task)
    end subroutine legion_task_get_local_args_f

    ! @see Legion::Task::local_arglen
    subroutine legion_task_get_local_arglen_f(task, local_arglen)
        implicit none

        type(legion_task_f_t), intent(in) :: task
        integer(c_size_t), intent(out)    :: local_arglen
        
        local_arglen =  legion_task_get_local_arglen_c(task)
    end subroutine legion_task_get_local_arglen_f

    ! @see Legion::Task::index_domain
    subroutine legion_task_get_index_domain_f(task, index_domain)
        implicit none
    
        type(legion_task_f_t), intent(in)    :: task
        type(legion_domain_f_t), intent(out) :: index_domain 
            
        index_domain = legion_task_get_index_domain_c(task)
    end subroutine legion_task_get_index_domain_f

    ! @see Legion::Task::regions
    subroutine legion_task_get_region_f(task, idx, rr)
        implicit none
        
        type(legion_task_f_t), intent(in)                :: task
        integer(c_int), intent(in)                       :: idx
        type(legion_region_requirement_f_t), intent(out) :: rr
            
        rr = legion_task_get_region_c(task, idx)
    end subroutine legion_task_get_region_f
    
    ! -----------------------------------------------------------------------
    ! Domain Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Domain::from_rect()
    subroutine legion_domain_from_rect_1d_f(r, domain)
        implicit none
    
        type(legion_rect_1d_f_t), intent(in) :: r
        type(legion_domain_f_t), intent(out) :: domain
            
        domain = legion_domain_from_rect_1d_c(r)
    end subroutine legion_domain_from_rect_1d_f

    ! @see Legion::Domain::from_rect()
    subroutine legion_domain_from_rect_2d_f(r, domain)
        implicit none
    
        type(legion_rect_2d_f_t), intent(in) :: r
        type(legion_domain_f_t), intent(out) :: domain
            
        domain = legion_domain_from_rect_2d_c(r)
    end subroutine legion_domain_from_rect_2d_f

    ! @see Legion::Domain::from_rect()
    subroutine legion_domain_from_rect_3d_f(r, domain)
        implicit none

        type(legion_rect_3d_f_t), intent(in) :: r
        type(legion_domain_f_t), intent(out) :: domain
            
        domain = legion_domain_from_rect_3d_c(r)
    end subroutine legion_domain_from_rect_3d_f

    ! @see Legion::Domain::get_rect()
    subroutine legion_domain_get_rect_1d_f(d, rect)
        implicit none

        type(legion_domain_f_t), intent(in)   :: d
        type(legion_rect_1d_f_t), intent(out) :: rect
            
        rect = legion_domain_get_rect_1d_c(d)
    end subroutine legion_domain_get_rect_1d_f

    ! @see Legion::Domain::get_rect()
    subroutine legion_domain_get_rect_2d_f(d, rect)
        implicit none

        type(legion_domain_f_t), intent(in)   :: d
        type(legion_rect_2d_f_t), intent(out) :: rect
            
        rect = legion_domain_get_rect_2d_c(d)
    end subroutine legion_domain_get_rect_2d_f

    ! @see Legion::Domain::get_rect()
    subroutine legion_domain_get_rect_3d_f(d, rect)
        implicit none

        type(legion_domain_f_t), intent(in)   :: d
        type(legion_rect_3d_f_t), intent(out) :: rect
            
        rect = legion_domain_get_rect_3d_c(d)
    end subroutine legion_domain_get_rect_3d_f

    ! @see Legion::Domain::get_volume()
    subroutine legion_domain_get_volume_f(d, volume)
        implicit none
    
        type(legion_domain_f_t), intent(in) :: d
        integer(c_size_t), intent(out)      :: volume
        
        volume = legion_domain_get_volume_c(d)
    end subroutine legion_domain_get_volume_f
    
    ! -----------------------------------------------------------------------
    ! Domain Point Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Domain::from_point()
    subroutine legion_domain_point_from_point_1d_f(p, domain_point)
        implicit none
    
        type(legion_point_1d_f_t), intent(in)      :: p
        type(legion_domain_point_f_t), intent(out) :: domain_point
            
        domain_point = legion_domain_point_from_point_1d_c(p)
    end subroutine legion_domain_point_from_point_1d_f

    ! @see Legion::Domain::from_point()
    subroutine legion_domain_point_from_point_2d_f(p, domain_point)
        implicit none
    
        type(legion_point_2d_f_t), intent(in)      :: p
        type(legion_domain_point_f_t), intent(out) :: domain_point
            
        domain_point = legion_domain_point_from_point_2d_c(p)
    end subroutine legion_domain_point_from_point_2d_f

    ! @see Legion::Domain::from_point()
    subroutine legion_domain_point_from_point_3d_f(p, domain_point)
        implicit none
    
        type(legion_point_3d_f_t), intent(in)      :: p
        type(legion_domain_point_f_t), intent(out) :: domain_point
            
        domain_point = legion_domain_point_from_point_3d_c(p)
    end subroutine legion_domain_point_from_point_3d_f
    
    ! -----------------------------------------------------------------------
    ! Future Map Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::FutureMap::wait_all_results()
    subroutine legion_future_map_wait_all_results_f(handle)
        implicit none
    
        type(legion_future_map_f_t), intent(in) :: handle
        
        call legion_future_map_wait_all_results_c(handle)
    end subroutine legion_future_map_wait_all_results_f
    
    ! -----------------------------------------------------------------------
    ! Index Space Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_index_space(Context, Domain)
    subroutine legion_index_space_create_domain_f(runtime, ctx, domain, index_space)
        implicit none
    
        type(legion_runtime_f_t), intent(in)      :: runtime
        type(legion_context_f_t), intent(in)      :: ctx
        type(legion_domain_f_t), intent(in)       :: domain
        type(legion_index_space_f_t), intent(out) :: index_space
            
        index_space = legion_index_space_create_domain_c(runtime, ctx, domain)
    end subroutine legion_index_space_create_domain_f

    ! @see Legion::Runtime::destroy_index_space()
    subroutine legion_index_space_destroy_f(runtime, ctx, handle)
        implicit none
    
        type(legion_runtime_f_t), intent(in)      :: runtime
        type(legion_context_f_t), intent(in)      :: ctx
        type(legion_index_space_f_t), intent(in)  :: handle
            
        call legion_index_space_destroy_c(runtime, ctx, handle)
    end subroutine legion_index_space_destroy_f

    ! @see Legion::Runtime::get_index_space_domain()
    subroutine legion_index_space_get_domain_f(runtime, handle, domain)
        implicit none

        type(legion_runtime_f_t), intent(in)     :: runtime
        type(legion_index_space_f_t), intent(in) :: handle
        type(legion_domain_f_t), intent(out)     :: domain
            
        domain = legion_index_space_get_domain_c(runtime, handle)
    end subroutine legion_index_space_get_domain_f

    ! @see Legion::Runtime::attach_name()
    subroutine legion_index_space_attach_name_f(runtime, handle, name, is_mutable)
        implicit none
    
        type(legion_runtime_f_t), intent(in)     :: runtime
        type(legion_index_space_f_t), intent(in) :: handle
        character(len=*), target, intent(in)     :: name
        logical(c_bool), intent(in)              :: is_mutable
        
        call legion_index_space_attach_name_c(runtime, handle, c_loc(name), is_mutable)
    end subroutine legion_index_space_attach_name_f
    
    ! -----------------------------------------------------------------------
    ! Index Partition Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_equal_partition()
    subroutine legion_index_partition_create_equal_f(runtime, ctx, parent, color_space, granularity, color, index_partition)
        implicit none
    
        type(legion_runtime_f_t), intent(in)          :: runtime
        type(legion_context_f_t), intent(in)          :: ctx
        type(legion_index_space_f_t), intent(in)      :: parent
        type(legion_index_space_f_t), intent(in)      :: color_space
        integer(c_size_t), intent(in)                 :: granularity
        integer(c_int), intent(in)                    :: color
        type(legion_index_partition_f_t), intent(out) :: index_partition
            
        index_partition = legion_index_partition_create_equal_c(runtime, ctx, parent, color_space, granularity, color)
    end subroutine legion_index_partition_create_equal_f

    ! @see Legion::Runtime::attach_name()
    subroutine legion_index_partition_attach_name_f(runtime, handle, name, is_mutable) 
        implicit none
    
        type(legion_runtime_f_t), intent(in)         :: runtime
        type(legion_index_partition_f_t), intent(in) :: handle
        character(len=*), target, intent(in)         :: name
        logical(c_bool), intent(in)                  :: is_mutable
        
        call legion_index_partition_attach_name_c(runtime, handle, c_loc(name), is_mutable) 
    end subroutine legion_index_partition_attach_name_f
    
    ! -----------------------------------------------------------------------
    ! Logical Region Tree Traversal Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::get_logical_partition()
    subroutine legion_logical_partition_create_f(runtime, ctx, parent, handle, logical_partition)
        implicit none
    
        type(legion_runtime_f_t), intent(in)            :: runtime
        type(legion_context_f_t), intent(in)            :: ctx
        type(legion_logical_region_f_t), intent(in)     :: parent
        type(legion_index_partition_f_t), intent(in)    :: handle
        type(legion_logical_partition_f_t), intent(out) :: logical_partition
            
        logical_partition = legion_logical_partition_create_c(runtime, ctx, parent, handle)
    end subroutine legion_logical_partition_create_f

    ! @see Legion::Runtime::attach_name()
    subroutine legion_logical_partition_attach_name_f(runtime, handle, name, is_mutable)
        implicit none
    
        type(legion_runtime_f_t), intent(in)           :: runtime
        type(legion_logical_partition_f_t), intent(in) :: handle
        character(len=*), target, intent(in)           :: name
        logical(c_bool), intent(in)                    :: is_mutable
        
        call legion_logical_partition_attach_name_c(runtime, handle, c_loc(name), is_mutable)
    end subroutine legion_logical_partition_attach_name_f
    
    ! -----------------------------------------------------------------------
    ! Field Space Operatiins 
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_field_space()
    subroutine legion_field_space_create_f(runtime, ctx, field_space)
        implicit none
    
        type(legion_runtime_f_t), intent(in)      :: runtime
        type(legion_context_f_t), intent(in)      :: ctx
        type(legion_field_space_f_t), intent(out) :: field_space
            
        field_space = legion_field_space_create_c(runtime, ctx)
    end subroutine legion_field_space_create_f

    ! @see Legion::Runtime::destroy_field_space()
    subroutine legion_field_space_destroy_f(runtime, ctx, handle)
        implicit none
    
        type(legion_runtime_f_t), intent(in)     :: runtime
        type(legion_context_f_t), intent(in)     :: ctx
        type(legion_field_space_f_t), intent(in) :: handle
            
        call legion_field_space_destroy_c(runtime, ctx, handle)
    end subroutine legion_field_space_destroy_f

    ! -----------------------------------------------------------------------
    ! Field Allocator 
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_field_allocator()
    subroutine legion_field_allocator_create_f(runtime, ctx, handle, field_allocator)
        implicit none
    
        type(legion_runtime_f_t), intent(in)          :: runtime
        type(legion_context_f_t), intent(in)          :: ctx
        type(legion_field_space_f_t), intent(in)      :: handle
        type(legion_field_allocator_f_t), intent(out) :: field_allocator
            
        field_allocator = legion_field_allocator_create_c(runtime, ctx, handle)
    end subroutine legion_field_allocator_create_f

    ! @see Legion::FieldAllocator::~FieldAllocator()
    subroutine legion_field_allocator_destroy_f(handle)
        implicit none
    
        type(legion_field_allocator_f_t), intent(in) :: handle
        
        call legion_field_allocator_destroy_c(handle)
    end subroutine legion_field_allocator_destroy_f

    ! @see Legion::FieldAllocator::allocate_field()
    subroutine legion_field_allocator_allocate_field_f(allocator, field_size, desired_fieldid, field_id)
        implicit none
    
        type(legion_field_allocator_f_t), intent(in) :: allocator                                          
        integer(c_size_t), intent(in)                :: field_size                                         
        integer(c_int), intent(in)                   :: desired_fieldid
        integer(c_int), intent(out)                  :: field_id
        
        field_id = legion_field_allocator_allocate_field_c(allocator, field_size, desired_fieldid)                                     
    end subroutine legion_field_allocator_allocate_field_f
    
    ! -----------------------------------------------------------------------
    ! Logical Region
    ! -----------------------------------------------------------------------
    ! @see Legion::Runtime::create_logical_region()
    subroutine legion_logical_region_create_f(runtime, ctx, index, fields, logical_region)
        implicit none
    
        type(legion_runtime_f_t), intent(in)         :: runtime
        type(legion_context_f_t), intent(in)         :: ctx
        type(legion_index_space_f_t), intent(in)     :: index
        type(legion_field_space_f_t), intent(in)     :: fields   
        type(legion_logical_region_f_t), intent(out) :: logical_region 
            
        logical_region = legion_logical_region_create_c(runtime, ctx, index, fields)    
    end subroutine legion_logical_region_create_f

    ! @see Legion::Runtime::destroy_logical_region()
    subroutine legion_logical_region_destroy_f(runtime, ctx, handle)
        implicit none
    
        type(legion_runtime_f_t), intent(in)        :: runtime
        type(legion_context_f_t), intent(in)        :: ctx
        type(legion_logical_region_f_t), intent(in) :: handle
            
        call legion_logical_region_destroy_c(runtime, ctx, handle)
    end subroutine legion_logical_region_destroy_f

    ! @see Legion::LogicalRegion::get_index_space
    subroutine legion_logical_region_get_index_space_f(handle, index_space)
        implicit none
    
        type(legion_logical_region_f_t), intent(in) :: handle
        type(legion_index_space_f_t), intent(out)   :: index_space
            
        index_space = legion_logical_region_get_index_space_c(handle)
    end subroutine legion_logical_region_get_index_space_f
    
    ! -----------------------------------------------------------------------
    ! Region Requirement Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::RegionRequirement::region
    subroutine legion_region_requirement_get_region_f(handle, logical_region)
        implicit none
        
        type(legion_region_requirement_f_t), intent(in) :: handle
        type(legion_logical_region_f_t), intent(out) :: logical_region
            
        logical_region = legion_region_requirement_get_region_c(handle)
    end subroutine legion_region_requirement_get_region_f
    
    ! @see Legion::RegionRequirement::privilege_fields
    subroutine legion_region_requirement_get_privilege_field_f(handle, idx, field_id)
        implicit none
    
        type(legion_region_requirement_f_t), intent(in) :: handle
        integer(c_int), intent(in)                      :: idx
        integer(c_int), intent(out)                     :: field_id
        
        field_id = legion_region_requirement_get_privilege_field_c(handle, idx)
    end subroutine legion_region_requirement_get_privilege_field_f
    
    ! -----------------------------------------------------------------------
    ! Physical Data Operations
    ! -----------------------------------------------------------------------
    subroutine legion_get_physical_region_by_id_f(regionptr, id, num_regions, physical_region)
        implicit none
    
        type(c_ptr), intent(in)                       :: regionptr
        integer(c_int), intent(in)                    :: id
        integer(c_int), intent(in)                    :: num_regions
        type(legion_physical_region_f_t), intent(out) :: physical_region
        
        physical_region = legion_get_physical_region_by_id_c(regionptr, id, num_regions)
    end subroutine legion_get_physical_region_by_id_f

    ! @see Legion::PhysicalRegion::get_field_accessor()
    subroutine legion_physical_region_get_field_accessor_array_1d_f(handle, fid, accessor)
        implicit none
    
        type(legion_physical_region_f_t), intent(in)    :: handle
        integer(c_int), intent(in)                      :: fid
        type(legion_accessor_array_1d_f_t), intent(out) :: accessor
            
        accessor = legion_physical_region_get_field_accessor_array_1d_c(handle, fid)
    end subroutine legion_physical_region_get_field_accessor_array_1d_f

    ! @see Legion::PhysicalRegion::get_field_accessor()
    subroutine legion_physical_region_get_field_accessor_array_2d_f(handle, fid, accessor)
        implicit none
    
        type(legion_physical_region_f_t), intent(in)    :: handle
        integer(c_int), intent(in)                      :: fid
        type(legion_accessor_array_2d_f_t), intent(out) :: accessor
            
        accessor = legion_physical_region_get_field_accessor_array_2d_c(handle, fid)
    end subroutine legion_physical_region_get_field_accessor_array_2d_f

    ! @see Legion::PhysicalRegion::get_field_accessor()
    subroutine legion_physical_region_get_field_accessor_array_3d_f(handle, fid, accessor)
        implicit none
    
        type(legion_physical_region_f_t), intent(in)    :: handle
        integer(c_int), intent(in)                      :: fid
        type(legion_accessor_array_3d_f_t), intent(out) :: accessor
            
        accessor = legion_physical_region_get_field_accessor_array_3d_c(handle, fid)
    end subroutine legion_physical_region_get_field_accessor_array_3d_f
    
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_raw_rect_ptr_f(handle, rect, subrect, offset, raw_ptr)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_rect_1d_f_t), intent(in)           :: rect
        type(legion_rect_1d_f_t), intent(out)          :: subrect ! pass reference
        type(legion_byte_offset_f_t), intent(out)      :: offset  ! pass reference
        type(c_ptr), intent(out)                       :: raw_ptr
            
        raw_ptr = legion_accessor_array_1d_raw_rect_ptr_c(handle, rect, subrect, offset)
    end subroutine legion_accessor_array_1d_raw_rect_ptr_f

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_2d_raw_rect_ptr_f(handle, rect, subrect, offset, raw_ptr)
        implicit none
    
        type(legion_accessor_array_2d_f_t), intent(in) :: handle
        type(legion_rect_2d_f_t), intent(in)           :: rect
        type(legion_rect_2d_f_t), intent(out)          :: subrect ! pass reference
        type(legion_byte_offset_f_t), intent(out)      :: offset(2)  ! pass reference
        type(c_ptr), intent(out)                       :: raw_ptr
            
        raw_ptr = legion_accessor_array_2d_raw_rect_ptr_c(handle, rect, subrect, offset)
    end subroutine legion_accessor_array_2d_raw_rect_ptr_f

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_3d_raw_rect_ptr_f(handle, rect, subrect, offset, raw_ptr)
        implicit none
    
        type(legion_accessor_array_3d_f_t), intent(in) :: handle
        type(legion_rect_3d_f_t), intent(in)           :: rect
        type(legion_rect_3d_f_t), intent(out)          :: subrect ! pass reference
        type(legion_byte_offset_f_t), intent(out)      :: offset(3)  ! pass reference
        type(c_ptr), intent(out)                       :: raw_ptr
            
        raw_ptr = legion_accessor_array_3d_raw_rect_ptr_c(handle, rect, subrect, offset)
    end subroutine legion_accessor_array_3d_raw_rect_ptr_f

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_ptr_f(handle, point, dst, bytes)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        type(c_ptr), intent(out)                       :: dst 
        integer(c_size_t), intent(in)                  :: bytes
        
        call legion_accessor_array_1d_read_point_c(handle, point, dst, bytes)
    end subroutine legion_accessor_array_1d_read_point_ptr_f
    
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_integer2_f(handle, point, dst)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        integer(kind=2), target, intent(out)           :: dst 
        
        call legion_accessor_array_1d_read_point_c(handle, point, c_loc(dst), c_sizeof(dst))
    end subroutine legion_accessor_array_1d_read_point_integer2_f
    
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_integer4_f(handle, point, dst)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        integer(kind=4), target, intent(out)           :: dst 
        
        call legion_accessor_array_1d_read_point_c(handle, point, c_loc(dst), c_sizeof(dst))
    end subroutine legion_accessor_array_1d_read_point_integer4_f
    
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_integer8_f(handle, point, dst)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        integer(kind=8), target, intent(out)           :: dst 
        
        call legion_accessor_array_1d_read_point_c(handle, point, c_loc(dst), c_sizeof(dst))
    end subroutine legion_accessor_array_1d_read_point_integer8_f
    
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_real4_f(handle, point, dst)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        real(kind=4), target, intent(out)              :: dst 
        
        call legion_accessor_array_1d_read_point_c(handle, point, c_loc(dst), c_sizeof(dst))
    end subroutine legion_accessor_array_1d_read_point_real4_f
    
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_real8_f(handle, point, dst)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        real(kind=8), target, intent(out)              :: dst 
        
        call legion_accessor_array_1d_read_point_c(handle, point, c_loc(dst), c_sizeof(dst))
    end subroutine legion_accessor_array_1d_read_point_real8_f
    
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_complex4_f(handle, point, dst)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        complex(kind=4), target, intent(out)           :: dst 
        
        call legion_accessor_array_1d_read_point_c(handle, point, c_loc(dst), c_sizeof(dst))
    end subroutine legion_accessor_array_1d_read_point_complex4_f
    
    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_read_point_complex8_f(handle, point, dst)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        complex(kind=8), target, intent(out)           :: dst 
        
        call legion_accessor_array_1d_read_point_c(handle, point, c_loc(dst), c_sizeof(dst))
    end subroutine legion_accessor_array_1d_read_point_complex8_f

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_2d_read_point_f(handle, point, dst, bytes)
        implicit none
    
        type(legion_accessor_array_2d_f_t), intent(in) :: handle
        type(legion_point_2d_f_t), intent(in)          :: point
        type(c_ptr), intent(out)                       :: dst
        integer(c_size_t), intent(in)                  :: bytes
        
        call legion_accessor_array_2d_read_point_c(handle, point, dst, bytes)
    end subroutine legion_accessor_array_2d_read_point_f

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_3d_read_point_f(handle, point, dst, bytes)
        implicit none
    
        type(legion_accessor_array_3d_f_t), intent(in) :: handle
        type(legion_point_3d_f_t), intent(in)          :: point
        type(c_ptr), intent(out)                       :: dst 
        integer(c_size_t), intent(in)                  :: bytes
        
        call legion_accessor_array_3d_read_point_c(handle, point, dst, bytes)
    end subroutine legion_accessor_array_3d_read_point_f

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_1d_write_point_f(handle, point, src, bytes)
        implicit none
    
        type(legion_accessor_array_1d_f_t), intent(in) :: handle
        type(legion_point_1d_f_t), intent(in)          :: point
        type(c_ptr), intent(in)                        :: src
        integer(c_size_t), intent(in)                  :: bytes
        
        call legion_accessor_array_1d_write_point_c(handle, point, src, bytes)
    end subroutine legion_accessor_array_1d_write_point_f

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_2d_write_point_f(handle, point, src, bytes)
        implicit none
    
        type(legion_accessor_array_2d_f_t), intent(in) :: handle
        type(legion_point_2d_f_t), intent(in)          :: point
        type(c_ptr), intent(in)                        :: src
        integer(c_size_t), intent(in)                  :: bytes
        
        call legion_accessor_array_2d_write_point_c(handle, point, src, bytes)
    end subroutine legion_accessor_array_2d_write_point_f 

    ! @see Legion::UnsafeFieldAccessor::ptr
    subroutine legion_accessor_array_3d_write_point_f(handle, point, src, bytes)
        implicit none
    
        type(legion_accessor_array_3d_f_t), intent(in) :: handle
        type(legion_point_3d_f_t), intent(in)          :: point
        type(c_ptr), intent(in)                        :: src
        integer(c_size_t), intent(in)                  :: bytes
        
        call legion_accessor_array_3d_write_point_c(handle, point, src, bytes)
    end subroutine legion_accessor_array_3d_write_point_f
    
    ! -----------------------------------------------------------------------
    ! File Operations
    ! -----------------------------------------------------------------------
    subroutine legion_field_map_create_f(field_map)
        implicit none
        
        type(legion_field_map_f_t), intent(out) :: field_map

        field_map = legion_field_map_create_c()
    end subroutine legion_field_map_create_f
    
    subroutine legion_field_map_destroy_f(handle)
        implicit none
        
        type(legion_field_map_f_t), intent(in) :: handle

        call legion_field_map_destroy_c(handle)
    end subroutine legion_field_map_destroy_f
    
    subroutine legion_field_map_insert_f(handle, key, value)
        implicit none
        
        type(legion_field_map_f_t), intent(in) :: handle
        integer(c_int), intent(in)             :: key
        character(len=*), target, intent(in)   :: value

        call legion_field_map_insert_c(handle, key, c_loc(value))
    end subroutine legion_field_map_insert_f
    
    ! @see Legion::Runtime::attach_hdf5()
    subroutine legion_runtime_attach_hdf5_f(runtime, ctx, filename, handle, parent, field_map, mode, physical_region)
        implicit none
        
        type(legion_runtime_f_t), intent(in)          :: runtime
        type(legion_context_f_t), intent(in)          :: ctx
        character(len=*), target, intent(in)          :: filename
        type(legion_logical_region_f_t), intent(in)   :: handle
        type(legion_logical_region_f_t), intent(in)   :: parent
        type(legion_field_map_f_t), intent(in)        :: field_map
        integer(c_int), intent(in)                    :: mode
        type(legion_physical_region_f_t), intent(out) :: physical_region

        physical_region = legion_runtime_attach_hdf5_c(runtime, ctx, c_loc(filename), handle, parent, field_map, mode)
    end subroutine legion_runtime_attach_hdf5_f
    
    ! @see Legion::Runtime::detach_hdf5()
    subroutine legion_runtime_detach_hdf5_f(runtime, ctx, region)
        implicit none
        
        type(legion_runtime_f_t), intent(in)         :: runtime
        type(legion_context_f_t), intent(in)         :: ctx
        type(legion_physical_region_f_t), intent(in) :: region

        call legion_runtime_detach_hdf5_c(runtime, ctx, region)
    end subroutine legion_runtime_detach_hdf5_f
    
    ! -----------------------------------------------------------------------
    ! Copy Operations
    ! -----------------------------------------------------------------------
    ! @see Legion::CopyLauncher::CopyLauncher()
    subroutine legion_copy_launcher_create_f(pred, id, launcher_tag, copy_launcher)
        implicit none
        
        type(legion_predicate_f_t), intent(in) :: pred
        integer(c_int), intent(in)             :: id
        integer(c_long), intent(in)            :: launcher_tag
        type(legion_copy_launcher_f_t), intent(out)   :: copy_launcher 
    
        copy_launcher = legion_copy_launcher_create_c(pred, id, launcher_tag)
    end subroutine legion_copy_launcher_create_f
    
    ! @see Legion::CopyLauncher::~CopyLauncher()
    subroutine legion_copy_launcher_destroy_f(handle)
        implicit none
        
        type(legion_copy_launcher_f_t), intent(in) :: handle

        call legion_copy_launcher_destroy_c(handle)
    end subroutine legion_copy_launcher_destroy_f
    
    ! @see Legion::Runtime::issue_copy_operation()
    subroutine legion_copy_launcher_execute_f(runtime, ctx, launcher)
        implicit none
        
        type(legion_runtime_f_t), intent(in)       :: runtime
        type(legion_context_f_t), intent(in)       :: ctx
        type(legion_copy_launcher_f_t), intent(in) :: launcher

        call legion_copy_launcher_execute_c(runtime, ctx, launcher)
    end subroutine legion_copy_launcher_execute_f
    
    ! @see Legion::CopyLauncher::add_copy_requirements()
    subroutine legion_copy_launcher_add_src_region_requirement_lr_f(launcher, handle, priv, prop, &
            parent, tag, verified, rr_idx)
        implicit none
        
        type(legion_copy_launcher_f_t), intent(in)  :: launcher
        type(legion_logical_region_f_t), intent(in) :: handle
        integer(c_int), intent(in)                  :: priv
        integer(c_int), intent(in)                  :: prop
        type(legion_logical_region_f_t), intent(in) :: parent
        integer(c_long), intent(in)                 :: tag
        logical(c_bool), intent(in)                 :: verified
        integer(c_int), intent(out)                 :: rr_idx

        rr_idx = legion_copy_launcher_add_src_region_requirement_lr_c(launcher, handle, priv, prop, parent, tag, verified)
    end subroutine legion_copy_launcher_add_src_region_requirement_lr_f
    
    ! @see Legion::CopyLauncher::add_copy_requirements()
    subroutine legion_copy_launcher_add_dst_region_requirement_lr_f(launcher, handle, priv, prop, &
            parent, tag, verified, rr_idx)
        implicit none
        
        type(legion_copy_launcher_f_t), intent(in)  :: launcher
        type(legion_logical_region_f_t), intent(in) :: handle
        integer(c_int), intent(in)                  :: priv
        integer(c_int), intent(in)                  :: prop
        type(legion_logical_region_f_t), intent(in) :: parent
        integer(c_long), intent(in)                 :: tag
        logical(c_bool), intent(in)                 :: verified
        integer(c_int), intent(out)                 :: rr_idx

        rr_idx = legion_copy_launcher_add_dst_region_requirement_lr_c(launcher, handle, priv, prop, parent, tag, verified)
    end subroutine legion_copy_launcher_add_dst_region_requirement_lr_f
    
    ! @see Legion::CopyLauncher::add_src_field()
    subroutine legion_copy_launcher_add_src_field_f(launcher, idx, fid, inst)
        implicit none
        
        type(legion_copy_launcher_f_t), intent(in) :: launcher
        integer(c_int), intent(in)                 :: idx
        integer(c_int), intent(in)                 :: fid
        logical(c_bool), intent(in)                :: inst

        call legion_copy_launcher_add_src_field_c(launcher, idx, fid, inst)
    end subroutine legion_copy_launcher_add_src_field_f
    
    ! @see Legion::CopyLauncher::add_dst_field()
    subroutine legion_copy_launcher_add_dst_field_f(launcher, idx, fid, inst)
        implicit none
        
        type(legion_copy_launcher_f_t), intent(in) :: launcher
        integer(c_int), intent(in)                 :: idx
        integer(c_int), intent(in)                 :: fid
        logical(c_bool), intent(in)                :: inst

        call legion_copy_launcher_add_dst_field_c(launcher, idx, fid, inst)
    end subroutine legion_copy_launcher_add_dst_field_f
    
    ! -----------------------------------------------------------------------
    ! Combined Operations
    ! -----------------------------------------------------------------------
    subroutine legion_task_get_index_space_from_logical_region_f(handle, tid, index_space)
        implicit none
    
        type(legion_task_f_t), intent(in)         :: handle
        integer(c_int), intent(in)                :: tid
        type(legion_index_space_f_t), intent(out) :: index_space
            
        !index_space = legion_task_get_index_space_from_logical_region_c(handle, tid)
        type(legion_region_requirement_f_t) :: rr
        type(legion_logical_region_f_t)     :: lr
            
        call legion_task_get_region_f(handle, tid, rr)
        call legion_region_requirement_get_region_f(rr, lr)
        call legion_logical_region_get_index_space_f(lr, index_space)
    end subroutine legion_task_get_index_space_from_logical_region_f
end module legion_fortran