module legion_fortran
    use, intrinsic :: iso_c_binding
    implicit none
    ! legion_privilege_mode_t
    integer(c_int), parameter :: NO_ACCESS = Z'00000000'
    integer(c_int), parameter :: READ_ONLY = Z'00000001'
    integer(c_int), parameter :: READ_WRITE = Z'00000007'
    integer(c_int), parameter :: WRITE_ONLY = Z'00000002'
    integer(c_int), parameter :: WRITE_DISCARD = Z'00000002'
    integer(c_int), parameter :: REDUCE = Z'00000004'
    
    ! legion_coherence_property_t
    integer(c_int), parameter :: EXCLUSIVE = 0
    integer(c_int), parameter :: ATOMIC = 1
    integer(c_int), parameter :: SIMULTANEOUS = 2
    integer(c_int), parameter :: RELAXED = 3
    
    ! C NEW_OPAQUE_TYPE_F
#define NEW_OPAQUE_TYPE_F(T) type, bind(C) :: T; type(c_ptr) :: impl; end type T
    NEW_OPAQUE_TYPE_F(legion_runtime_f_t)
    NEW_OPAQUE_TYPE_F(legion_context_f_t)
    NEW_OPAQUE_TYPE_F(legion_domain_point_iterator_f_t)
    NEW_OPAQUE_TYPE_F(legion_coloring_f_t)
    NEW_OPAQUE_TYPE_F(legion_domain_coloring_f_t)
    NEW_OPAQUE_TYPE_F(legion_point_coloring_f_t)
    NEW_OPAQUE_TYPE_F(legion_domain_point_coloring_f_t)
    NEW_OPAQUE_TYPE_F(legion_multi_domain_point_coloring_f_t)
    NEW_OPAQUE_TYPE_F(legion_index_space_allocator_f_t)
    NEW_OPAQUE_TYPE_F(legion_field_allocator_f_t)
    NEW_OPAQUE_TYPE_F(legion_argument_map_f_t)
    NEW_OPAQUE_TYPE_F(legion_predicate_f_t)
    NEW_OPAQUE_TYPE_F(legion_future_f_t)
    NEW_OPAQUE_TYPE_F(legion_future_map_f_t)
    NEW_OPAQUE_TYPE_F(legion_task_launcher_f_t)
    NEW_OPAQUE_TYPE_F(legion_index_launcher_f_t)
    NEW_OPAQUE_TYPE_F(legion_inline_launcher_f_t)
    NEW_OPAQUE_TYPE_F(legion_copy_launcher_f_t)
    NEW_OPAQUE_TYPE_F(legion_acquire_launcher_f_t)
    NEW_OPAQUE_TYPE_F(legion_release_launcher_f_t)
    NEW_OPAQUE_TYPE_F(legion_must_epoch_launcher_f_t)
    NEW_OPAQUE_TYPE_F(legion_physical_region_f_t)
    NEW_OPAQUE_TYPE_F(legion_accessor_array_1d_f_t)
    NEW_OPAQUE_TYPE_F(legion_accessor_array_2d_f_t)
    NEW_OPAQUE_TYPE_F(legion_accessor_array_3d_f_t)
    NEW_OPAQUE_TYPE_F(legion_index_iterator_f_t)
    NEW_OPAQUE_TYPE_F(legion_task_f_t)
    NEW_OPAQUE_TYPE_F(legion_inline_f_t)
    NEW_OPAQUE_TYPE_F(legion_mappable_f_t)
    NEW_OPAQUE_TYPE_F(legion_region_requirement_f_t)
    NEW_OPAQUE_TYPE_F(legion_machine_f_t)
    NEW_OPAQUE_TYPE_F(legion_mapper_f_t)
    NEW_OPAQUE_TYPE_F(legion_default_mapper_f_t)
    NEW_OPAQUE_TYPE_F(legion_processor_query_f_t)
    NEW_OPAQUE_TYPE_F(legion_memory_query_f_t)
    NEW_OPAQUE_TYPE_F(legion_machine_query_interface_f_t)
    NEW_OPAQUE_TYPE_F(legion_execution_constraint_set_f_t)
    NEW_OPAQUE_TYPE_F(legion_layout_constraint_set_f_t)
    NEW_OPAQUE_TYPE_F(legion_task_layout_constraint_set_f_t)
    NEW_OPAQUE_TYPE_F(legion_slice_task_output_f_t)
    NEW_OPAQUE_TYPE_F(legion_map_task_input_f_t)
    NEW_OPAQUE_TYPE_F(legion_map_task_output_f_t)
    NEW_OPAQUE_TYPE_F(legion_physical_instance_f_t)
    NEW_OPAQUE_TYPE_F(legion_mapper_runtime_f_t)
    NEW_OPAQUE_TYPE_F(legion_mapper_context_f_t)
    NEW_OPAQUE_TYPE_F(legion_field_map_f_t)
#undef NEW_OPAQUE_TYPE_F

    ! point 1d, 2d, 3d
#define NEW_POINT_TYPE_F(T, DIM) type, bind(C) :: T; integer(c_long_long), dimension(0:DIM-1) :: x; end type T
    NEW_POINT_TYPE_F(legion_point_1d_f_t, 1)
    NEW_POINT_TYPE_F(legion_point_2d_f_t, 2)
    NEW_POINT_TYPE_F(legion_point_3d_f_t, 3)
#undef NEW_POINT_TYPE_F

    ! rect 1d, 2d, 3d
#define NEW_RECT_TYPE_F(T, PT) type, bind(C) :: T; type(PT) :: lo, hi; end type T
    NEW_RECT_TYPE_F(legion_rect_1d_f_t, legion_point_1d_f_t)
    NEW_RECT_TYPE_F(legion_rect_2d_f_t, legion_point_2d_f_t)
    NEW_RECT_TYPE_F(legion_rect_3d_f_t, legion_point_3d_f_t)
#undef NEW_RECT_TYPE_F

    ! Legion::Domain
    type, bind(C) :: legion_domain_f_t
        integer(c_long_long)                                  :: is_id
        integer(c_int)                                        :: dim
        ! check MAX_DOMAIN_DIM = 2 * REALM_MAX_RECT_DIM
#define MAX_DOMAIN_DIM_F 6 
        integer(c_long_long), dimension(0:MAX_DOMAIN_DIM_F-1) :: rect_data
#undef MAX_DOMAIN_DIM_F        
    end type legion_domain_f_t
    
    ! Legion::DomainPoint
    type, bind(C) :: legion_domain_point_f_t
        integer(c_int)                                        :: dim
#define MAX_POINT_DIM_F 6
        integer(c_long_long), dimension(0:MAX_POINT_DIM_F-1) :: point_data
#undef  MAX_POINT_DIM_F
    end type legion_domain_point_f_t
    
    ! Legion::IndexSpace
    type, bind(C) :: legion_index_space_f_t
        integer(c_int) :: id
        integer(c_int) :: tid
        integer(c_int) :: type_tag
    end type legion_index_space_f_t
    
    ! Legion::IndexPartition
    type, bind(C) :: legion_index_partition_f_t
        integer(c_int) :: id
        integer(c_int) :: tid
        integer(c_int) :: type_tag
    end type legion_index_partition_f_t
    
    ! Legion::FieldSpace
    type, bind(C) :: legion_field_space_f_t
        integer(c_int) :: id
    end type legion_field_space_f_t
    
    ! Legion::LogicalRegion
    type, bind(C) :: legion_logical_region_f_t
        integer(c_int)               :: tree_id
        type(legion_index_space_f_t) :: index_space
        type(legion_field_space_f_t) :: field_space
    end type legion_logical_region_f_t
     
    ! Legion::LogicalPartition
    type, bind(C) :: legion_logical_partition_f_t
        integer(c_int)                   :: tree_id
        type(legion_index_partition_f_t) :: index_partition
        type(legion_field_space_f_t)     :: field_space
    end type legion_logical_partition_f_t
    
    ! Legion::TaskConfigOptions
    type, bind(C) :: legion_task_config_options_f_t
        logical(c_bool) :: leaf
        logical(c_bool) :: inner
        logical(c_bool) :: idempotent
    end type legion_task_config_options_f_t
    
    ! Legion::TaskArgument
    type, bind(C) :: legion_task_argument_f_t
        type(c_ptr)         :: args
        integer(c_size_t)   :: arglen
    end type legion_task_argument_f_t
    
    ! offest
    type, bind(C) :: legion_byte_offset_f_t
        integer(c_int) :: offset
    end type legion_byte_offset_f_t
    
    ! C typedef enum
  !  enum, bind(C) :: legion_processor_kind_t
   !     enumrator :: NO_KIND = 0
    !    TOC_PROC, LOC_PROC, UTIL_PROC, IO_PROC, PROC_GROUP, PROC_SET, OMP_PROC
    !end enum

    interface
        ! -----------------------------------------------------------------------
        ! Start-up Operations
        ! -----------------------------------------------------------------------
        ! Legion::Runtime::set_top_level_task_id()
        subroutine legion_runtime_set_top_level_task_id_f(top_id) &
                       bind(C, name="legion_runtime_set_top_level_task_id")
            use iso_c_binding
            implicit none
            
            integer(c_int), value, intent(in) :: top_id
        end subroutine legion_runtime_set_top_level_task_id_f
        
        ! Legion::ExecutionConstraintSet::ExecutionConstraintSet()
        function legion_execution_constraint_set_create_f() &
                     bind(C, name="legion_execution_constraint_set_create")
            use iso_c_binding
            import legion_execution_constraint_set_f_t
            implicit none
            
            type(legion_execution_constraint_set_f_t) :: legion_execution_constraint_set_create_f
        end function
        
        ! Legion::ExecutionConstraintSet::add_constraint(Legion::ProcessorConstraint)
        subroutine legion_execution_constraint_set_add_processor_constraint_f(handle, proc_kind) &
                       bind(C, name="legion_execution_constraint_set_add_processor_constraint")
            use iso_c_binding
            import legion_execution_constraint_set_f_t
            implicit none
            
            type(legion_execution_constraint_set_f_t), value, intent(in)    :: handle
            integer(c_int), value, intent(in)                               :: proc_kind
        end subroutine
        
        ! Legion::TaskLayoutConstraintSet::TaskLayoutConstraintSet()
        function legion_task_layout_constraint_set_create_f() &
                     bind(C, name="legion_task_layout_constraint_set_create")
            use iso_c_binding
            import legion_task_layout_constraint_set_f_t
            implicit none
            
            type(legion_task_layout_constraint_set_f_t) :: legion_task_layout_constraint_set_create_f
        end function
        
        ! Legion::Runtime::preregister_task_variant()
        function legion_runtime_preregister_task_variant_fnptr_f(id, task_name, &
                                                                 execution_constraints, &
                                                                 layout_constraints, &
                                                                 options, &
                                                                 wrapped_task_pointer, &
                                                                 userdata, &
                                                                 userlen) &
                     bind(C, name="legion_runtime_preregister_task_variant_fnptr")
            use iso_c_binding
            import legion_execution_constraint_set_f_t
            import legion_task_layout_constraint_set_f_t
            import legion_task_config_options_f_t
            implicit none
            
            integer(c_int)                                                  :: legion_runtime_preregister_task_variant_fnptr_f
            character(kind=c_char), intent(in)                              :: task_name(*)
            integer(c_int), value, intent(in)                               :: id
            type(legion_execution_constraint_set_f_t), value, intent(in)    :: execution_constraints
            type(legion_task_layout_constraint_set_f_t), value, intent(in)  :: layout_constraints
            type(legion_task_config_options_f_t), value, intent(in)         :: options
            type(c_funptr), value, intent(in)                               :: wrapped_task_pointer
            type(c_ptr), value, intent(in)                                  :: userdata
            integer(c_size_t), value, intent(in)                            :: userlen
        end function
        
        ! Legion::Runtime::start()
        function legion_runtime_start_f(argc, argv, background) &
                     bind(C, name="legion_runtime_start")
            use iso_c_binding
            implicit none
            
            integer(c_int)                      :: legion_runtime_start_f
            integer(c_int), value, intent(in)   :: argc
            type(c_ptr), value, intent(in)      :: argv
            logical(c_bool), value, intent(in)  :: background
        end function
        
        ! Legion::LegionTaskWrapper::legion_task_preamble()
        subroutine legion_task_preamble_f(tdata, tdatalen, proc_id, &
                                          task, regionptr, num_regions, &
                                          ctx, runtime) &
                       bind(C, name="legion_task_preamble")
            use iso_c_binding
            import legion_task_f_t
            import legion_physical_region_f_t
            import legion_context_f_t
            import legion_runtime_f_t
            implicit none
            
            type(c_ptr), intent(in)                         :: tdata ! pass reference
            integer(c_size_t), value, intent(in)            :: tdatalen
            integer(c_long_long), value, intent(in)         :: proc_id
            type(legion_task_f_t), intent(out)              :: task ! pass reference
            type(c_ptr), intent(out)                        :: regionptr
            integer(c_int), intent(out)                     :: num_regions ! pass reference
            type(legion_context_f_t), intent(out)           :: ctx ! pass reference          
            type(legion_runtime_f_t), intent(out)           :: runtime ! pass reference
        end subroutine
        
        ! Legion::LegionTaskWrapper::legion_task_postamble()
        subroutine legion_task_postamble_f(runtime, ctx, retval, retsize) &
                       bind(C, name="legion_task_postamble")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            implicit none
            
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
            type(c_ptr), value, intent(in)              :: retval
            integer(c_size_t), value, intent(in)        :: retsize
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Task Launcher
        ! -----------------------------------------------------------------------
        ! @see Legion::TaskLauncher::TaskLauncher()
        function legion_task_launcher_create_f(tid, arg, pred, id, tag) &
                     bind(C, name="legion_task_launcher_create")
            use iso_c_binding
            import legion_task_launcher_f_t
            import legion_task_argument_f_t
            import legion_predicate_f_t
            implicit none
            
            type(legion_task_launcher_f_t)                      :: legion_task_launcher_create_f
            integer(c_int), value, intent(in)                   :: tid
            type(legion_task_argument_f_t), value, intent(in)   :: arg
            type(legion_predicate_f_t), value, intent(in)       :: pred
            integer(c_int), value, intent(in)                   :: id
            integer(c_long), value, intent(in)                  :: tag
        end function
        
        ! @see Legion::Runtime::execute_task()
        function legion_task_launcher_execute_f(runtime, ctx, launcher) &
                     bind(C, name="legion_task_launcher_execute")
            use iso_c_binding
            import legion_future_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_task_launcher_f_t
            implicit none
            
            type(legion_future_f_t)                            :: legion_task_launcher_execute_f
            type(legion_runtime_f_t), value, intent(in)        :: runtime
            type(legion_context_f_t), value, intent(in)        :: ctx
            type(legion_task_launcher_f_t), value, intent(in)  :: launcher
        end function
        
        ! @see Legion::TaskLauncher::add_region_requirement()
        function legion_task_launcher_add_region_requirement_logical_region_f(launcher, handle, priv, prop, parent, tag, verified) &
                     bind (C, name="legion_task_launcher_add_region_requirement_logical_region")
            use iso_c_binding
            import legion_task_launcher_f_t
            import legion_logical_region_f_t
            implicit none
            
            integer(c_int)                                     :: legion_task_launcher_add_region_requirement_logical_region_f
            type(legion_task_launcher_f_t), value, intent(in)  :: launcher
            type(legion_logical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                  :: priv
            integer(c_int), value, intent(in)                  :: prop
            type(legion_logical_region_f_t), value, intent(in) :: parent
            integer(c_long), value, intent(in)                 :: tag
            logical(c_bool), value, intent(in)                 :: verified
        end function
        
        ! @see Legion::TaskLaunchxer::add_field()
        subroutine legion_task_launcher_add_field_f(launcher, idx, fid, inst) &
                       bind(C, name="legion_task_launcher_add_field")
            use iso_c_binding
            import legion_task_launcher_f_t
            implicit none
            
            type(legion_task_launcher_f_t), value, intent(in) :: launcher
            integer(c_int), value, intent(in)                 :: idx
            integer(c_int), value, intent(in)                 :: fid
            logical(c_bool), value, intent(in)                :: inst 
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Index Launcher
        ! -----------------------------------------------------------------------
        ! @see Legion::IndexTaskLauncher::IndexTaskLauncher()
        function legion_index_launcher_create_f(tid, domain, global_arg, map, pred, must, id, tag) &
                     bind(C, name="legion_index_launcher_create")
            use iso_c_binding
            import legion_index_launcher_f_t
            import legion_domain_f_t
            import legion_task_argument_f_t
            import legion_argument_map_f_t
            import legion_predicate_f_t
            implicit none
            
            type(legion_index_launcher_f_t)                     :: legion_index_launcher_create_f
            integer(c_int), value, intent(in)                   :: tid
            type(legion_domain_f_t), value, intent(in)          :: domain
            type(legion_task_argument_f_t), value, intent(in)   :: global_arg
            type(legion_argument_map_f_t), value, intent(in)    :: map
            type(legion_predicate_f_t), value, intent(in)       :: pred
            logical(c_bool), value, intent(in)                  :: must
            integer(c_int), value, intent(in)                   :: id
            integer(c_long), value, intent(in)                  :: tag
        end function
        
        ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &)
        function legion_index_launcher_execute_f(runtime, ctx, launcher) &
                     bind(C, name="legion_index_launcher_execute")
            use iso_c_binding
            import legion_future_map_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_launcher_f_t
            implicit none
            
            type(legion_future_map_f_t)                         :: legion_index_launcher_execute_f
            type(legion_runtime_f_t), value, intent(in)         :: runtime
            type(legion_context_f_t), value, intent(in)         :: ctx
            type(legion_index_launcher_f_t), value, intent(in)  :: launcher
        end function
        
        ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, ReductionOpID)
        function legion_index_launcher_execute_reduction_f(runtime, ctx, launcher, redop) &
                     bind(C, name="legion_index_launcher_execute_reduction")
            use iso_c_binding
            import legion_future_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_launcher_f_t
            implicit none
            
            type(legion_future_f_t)                            :: legion_index_launcher_execute_reduction_f
            type(legion_runtime_f_t), value, intent(in)        :: runtime
            type(legion_context_f_t), value, intent(in)        :: ctx
            type(legion_index_launcher_f_t), value, intent(in) :: launcher
            integer(c_int), value, intent(in)                  :: redop 
        end function
        
        ! @see Legion::IndexTaskLauncher::add_region_requirement()
        function legion_index_launcher_add_region_requirement_lp_f(launcher, handle, proj, priv, &
                                                                   prop, parent, tag, verified) &
                     bind (C, name="legion_index_launcher_add_region_requirement_logical_partition")
            use iso_c_binding
            import legion_index_launcher_f_t
            import legion_logical_partition_f_t
            import legion_logical_region_f_t
            implicit none
            
            integer(c_int)                              :: legion_index_launcher_add_region_requirement_lp_f
            type(legion_index_launcher_f_t), value, intent(in)    :: launcher
            type(legion_logical_partition_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                     :: proj
            integer(c_int), value, intent(in)                     :: priv
            integer(c_int), value, intent(in)                     :: prop
            type(legion_logical_region_f_t), value, intent(in)    :: parent
            integer(c_long), value, intent(in)                    :: tag
            logical(c_bool), value, intent(in)                    :: verified
        end function
        
        ! @see Legion::TaskLaunchxer::add_field()
        subroutine legion_index_launcher_add_field_f(launcher, idx, fid, inst) &
                       bind(C, name="legion_index_launcher_add_field")
            use iso_c_binding
            import legion_index_launcher_f_t
            implicit none
            
            type(legion_index_launcher_f_t), value, intent(in) :: launcher
            integer(c_int), value, intent(in)                 :: idx
            integer(c_int), value, intent(in)                 :: fid
            logical(c_bool), value, intent(in)                :: inst 
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Predicate Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Predicate::TRUE_PRED
        function legion_predicate_true_f() &
                     bind(C, name="legion_predicate_true")
            use iso_c_binding
            import legion_predicate_f_t
            implicit none
            
            type(legion_predicate_f_t)  :: legion_predicate_true_f
        end function
        
        ! @see Legion::Predicate::FALSE_PRED
        function legion_predicate_false_f() &
                     bind(C, name="legion_predicate_false")
            use iso_c_binding
            import legion_predicate_f_t
            implicit none
            
            type(legion_predicate_f_t)  :: legion_predicate_false_f
        end function
        
        ! -----------------------------------------------------------------------
        ! Argument Map
        ! -----------------------------------------------------------------------
        ! @see Legion::ArgumentMap::ArgumentMap()
        function legion_argument_map_create_f() &
                     bind(C, name="legion_argument_map_create")
            use iso_c_binding
            import legion_argument_map_f_t
            implicit none
            
            type(legion_argument_map_f_t) :: legion_argument_map_create_f
        end function
        
        ! @see Legion::ArgumentMap::set_point()
        subroutine legion_argument_map_set_point_f(map, dp, arg, replace) &
                       bind(C, name="legion_argument_map_set_point")
            use iso_c_binding
            import legion_argument_map_f_t
            import legion_domain_point_f_t
            import legion_task_argument_f_t
            implicit none
            
            type(legion_argument_map_f_t), value, intent(in)  :: map
            type(legion_domain_point_f_t), value, intent(in)  :: dp 
            type(legion_task_argument_f_t), value, intent(in) :: arg
            logical(c_bool), value, intent(in)                :: replace 
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Task Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Task::args
        function legion_task_get_args_f(task) &
                     bind(C, name="legion_task_get_args")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            type(c_ptr)                              :: legion_task_get_args_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        ! @see Legion::Task::arglen
        function legion_task_get_arglen_f(task) &
                     bind(C, name="legion_task_get_arglen")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            integer(c_size_t)                        :: legion_task_get_arglen_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        ! @see Legion::Task::local_args
        function legion_task_get_local_args_f(task) &
                     bind(C, name="legion_task_get_local_args")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            type(c_ptr)                              :: legion_task_get_local_args_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        ! @see Legion::Task::local_arglen
        function legion_task_get_local_arglen_f(task) &
                     bind(C, name="legion_task_get_local_arglen")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            integer(c_size_t)                        :: legion_task_get_local_arglen_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        ! @see Legion::Task::index_domain
        function legion_task_get_index_domain_f(task) &
                     bind(C, name="legion_task_get_index_domain")
            use iso_c_binding
            import legion_domain_f_t
            import legion_task_f_t
            implicit none
            
            type(legion_domain_f_t)                  :: legion_task_get_index_domain_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        ! -----------------------------------------------------------------------
        ! Domain Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Domain::from_rect()
        function legion_domain_from_rect_1d_f(r) &
                     bind(C, name="legion_domain_from_rect_1d")
            use iso_c_binding
            import legion_rect_1d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_domain_f_t)                     :: legion_domain_from_rect_1d_f
            type(legion_rect_1d_f_t), value, intent(in) :: r
        end function
        
        ! @see Legion::Domain::from_rect()
        function legion_domain_from_rect_2d_f(r) &
                     bind(C, name="legion_domain_from_rect_2d")
            use iso_c_binding
            import legion_rect_2d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_domain_f_t)                     :: legion_domain_from_rect_2d_f
            type(legion_rect_2d_f_t), value, intent(in) :: r
        end function
        
        ! @see Legion::Domain::from_rect()
        function legion_domain_from_rect_3d_f(r) &
                     bind(C, name="legion_domain_from_rect_3d")
            use iso_c_binding
            import legion_rect_3d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_domain_f_t)                     :: legion_domain_from_rect_3d_f
            type(legion_rect_3d_f_t), value, intent(in) :: r
        end function
        
        ! @see Legion::Domain::get_rect()
        function legion_domain_get_rect_1d_f(d) &
                     bind(C, name="legion_domain_get_rect_1d")
            use iso_c_binding
            import legion_rect_1d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_rect_1d_f_t)                   :: legion_domain_get_rect_1d_f
            type(legion_domain_f_t), value, intent(in) :: d
        end function
        
        ! @see Legion::Domain::get_rect()
        function legion_domain_get_rect_2d_f(d) &
                     bind(C, name="legion_domain_get_rect_2d")
            use iso_c_binding
            import legion_rect_2d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_rect_2d_f_t)                   :: legion_domain_get_rect_2d_f
            type(legion_domain_f_t), value, intent(in) :: d
        end function
        
        ! @see Legion::Domain::get_rect()
        function legion_domain_get_rect_3d_f(d) &
                     bind(C, name="legion_domain_get_rect_3d")
            use iso_c_binding
            import legion_rect_3d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_rect_3d_f_t)                   :: legion_domain_get_rect_3d_f
            type(legion_domain_f_t), value, intent(in) :: d
        end function
        
        ! @see Legion::Domain::get_volume()
        function legion_domain_get_volume_f(d) &
                     bind(C, name="legion_domain_get_volume")
            use iso_c_binding
            import legion_domain_f_t
            implicit none
            
            integer(c_size_t)                          :: legion_domain_get_volume_f
            type(legion_domain_f_t), value, intent(in) :: d
        end function
        
        ! -----------------------------------------------------------------------
        ! Domain Point Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Domain::from_point()
        function legion_domain_point_from_point_1d_f(p) &
                     bind(C, name="legion_domain_point_from_point_1d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_1d_f_t
            implicit none
            
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_1d_f
            type(legion_point_1d_f_t), value, intent(in) :: p
        end function
        
        ! @see Legion::Domain::from_point()
        function legion_domain_point_from_point_2d_f(p) &
                     bind(C, name="legion_domain_point_from_point_2d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_2d_f_t
            implicit none
            
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_2d_f
            type(legion_point_2d_f_t), value, intent(in) :: p
        end function
        
        ! @see Legion::Domain::from_point()
        function legion_domain_point_from_point_3d_f(p) &
                     bind(C, name="legion_domain_point_from_point_3d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_3d_f_t
            implicit none
            
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_3d_f
            type(legion_point_3d_f_t), value, intent(in) :: p
        end function
        
        ! -----------------------------------------------------------------------
        ! Future Map Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::FutureMap::wait_all_results()
        subroutine legion_future_map_wait_all_results_f(handle) &
                       bind(C, name="legion_future_map_wait_all_results")
            use iso_c_binding
            import legion_future_map_f_t
            implicit none
            
            type(legion_future_map_f_t), value, intent(in) :: handle
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Index Space Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_index_space(Context, Domain)
        function legion_index_space_create_domain_f(runtime, ctx, domain) &
                     bind(C, name="legion_index_space_create_domain")
            use iso_c_binding
            import legion_index_space_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_index_space_f_t)                :: legion_index_space_create_domain_f
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
            type(legion_domain_f_t), value, intent(in)  :: domain
        end function
        
        ! @see Legion::Runtime::destroy_index_space()
        subroutine legion_index_space_destroy_f(runtime, ctx, handle) &
                       bind(C, name="legion_index_space_destroy")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_space_f_t
            implicit none
            
            type(legion_runtime_f_t), value, intent(in)      :: runtime
            type(legion_context_f_t), value, intent(in)      :: ctx
            type(legion_index_space_f_t), value, intent(in)  :: handle
        end subroutine
        
        ! @see Legion::Runtime::get_index_space_domain()
        function legion_index_space_get_domain_f(runtime, handle) &
                     bind(C, name="legion_index_space_get_domain")
            use iso_c_binding
            import legion_domain_f_t
            import legion_runtime_f_t
            import legion_index_space_f_t
            implicit none
            
            type(legion_domain_f_t)                         :: legion_index_space_get_domain_f
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_index_space_f_t), value, intent(in) :: handle
        end function
        
        ! @see Legion::Runtime::attach_name()
        subroutine legion_index_space_attach_name_f(runtime, handle, name, is_mutable) &
                       bind (C, name="legion_index_space_attach_name")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_index_space_f_t
            implicit none
            
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_index_space_f_t), value, intent(in) :: handle
            type(c_ptr), value, intent(in)                 :: name
            logical(c_bool), value, intent(in)              :: is_mutable
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Index Partition Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_equal_partition()
        function legion_index_partition_create_equal_f(runtime, ctx, parent, color_space, granularity, color) &
                     bind(C, name="legion_index_partition_create_equal")
            use iso_c_binding
            import legion_index_partition_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_space_f_t
            implicit none
            
            type(legion_index_partition_f_t)                :: legion_index_partition_create_equal_f
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_context_f_t), value, intent(in)     :: ctx
            type(legion_index_space_f_t), value, intent(in) :: parent
            type(legion_index_space_f_t), value, intent(in) :: color_space
            integer(c_size_t), value, intent(in)            :: granularity
            integer(c_int), value, intent(in)               :: color
        end function
        
        ! @see Legion::Runtime::attach_name()
        subroutine legion_index_partition_attach_name_f(runtime, handle, name, is_mutable) &
                       bind (C, name="legion_index_partition_attach_name")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_index_partition_f_t
            implicit none
            
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_index_partition_f_t), value, intent(in) :: handle
            type(c_ptr), value, intent(in)                 :: name
            logical(c_bool), value, intent(in)              :: is_mutable
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Logical Region Tree Traversal Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::get_logical_partition()
        function legion_logical_partition_create_f(runtime, ctx, parent, handle) &
                     bind (C, name="legion_logical_partition_create")
            use iso_c_binding
            import legion_logical_partition_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_logical_region_f_t
            import legion_index_partition_f_t
            implicit none
            
            type(legion_logical_partition_f_t) :: legion_logical_partition_create_f
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
            type(legion_logical_region_f_t), value, intent(in) :: parent
            type(legion_index_partition_f_t), value, intent(in) :: handle
        end function
        
        ! @see Legion::Runtime::attach_name()
        subroutine legion_logical_partition_attach_name_f(runtime, handle, name, is_mutable) &
                       bind (C, name="legion_logical_partition_attach_name")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_logical_partition_f_t
            implicit none
            
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_logical_partition_f_t), value, intent(in) :: handle
            type(c_ptr), value, intent(in)                 :: name
            logical(c_bool), value, intent(in)              :: is_mutable
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Field Space Operatiins 
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_field_space()
        function legion_field_space_create_f(runtime, ctx) &
                     bind(C, name="legion_field_space_create")
            use iso_c_binding
            import legion_field_space_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            implicit none
            
            type(legion_field_space_f_t)                :: legion_field_space_create_f
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
        end function
        
        ! @see Legion::Runtime::destroy_field_space()
        subroutine legion_field_space_destroy_f(runtime, ctx, handle) &
                       bind(C, name="legion_field_space_destroy")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_field_space_f_t
            implicit none
            
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_context_f_t), value, intent(in)     :: ctx
            type(legion_field_space_f_t), value, intent(in) :: handle
        end subroutine
        
        ! -----------------------------------------------------------------------
        ! Field Allocator 
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_field_allocator()
        function legion_field_allocator_create_f(runtime, ctx, handle) &
                     bind(C, name="legion_field_allocator_create")
            use iso_c_binding
            import legion_field_allocator_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_field_space_f_t
            implicit none
            
            type(legion_field_allocator_f_t)                :: legion_field_allocator_create_f
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_context_f_t), value, intent(in)     :: ctx
            type(legion_field_space_f_t), value, intent(in) :: handle
        end function
        
        ! @see Legion::FieldAllocator::~FieldAllocator()
        subroutine legion_field_allocator_destroy_f(handle) bind(C, name="legion_field_allocator_destroy")
            use iso_c_binding
            import legion_field_allocator_f_t
            implicit none
            
            type(legion_field_allocator_f_t), value, intent(in) :: handle
        end subroutine
        
        ! @see Legion::FieldAllocator::allocate_field()
        function legion_field_allocator_allocate_field_f(allocator, field_size, desired_fieldid) &
                                                         bind (C, name="legion_field_allocator_allocate_field")
            use iso_c_binding
            import legion_field_allocator_f_t
            implicit none
            
            integer(c_int)                                      :: legion_field_allocator_allocate_field_f
            type(legion_field_allocator_f_t), value, intent(in) :: allocator                                          
            integer(c_size_t), value, intent(in)                :: field_size                                         
            integer(c_int), value, intent(in)                   :: desired_fieldid                                        
        end function
        
        ! -----------------------------------------------------------------------
        ! Logical Region
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_logical_region()
        function legion_logical_region_create_f(runtime, ctx, index, fields) &
                     bind(C, name="legion_logical_region_create")
            use iso_c_binding
            import legion_logical_region_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_space_f_t
            import legion_field_space_f_t
            implicit none
            
            type(legion_logical_region_f_t)                 :: legion_logical_region_create_f
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_context_f_t), value, intent(in)     :: ctx
            type(legion_index_space_f_t), value, intent(in) :: index
            type(legion_field_space_f_t), value, intent(in) :: fields        
        end function
        
        ! @see Legion::Runtime::destroy_logical_region()
        subroutine legion_logical_region_destroy_f(runtime, ctx, handle) &
                       bind(C, name="legion_logical_region_destroy")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_logical_region_f_t
            implicit none
            
            type(legion_runtime_f_t), value, intent(in)        :: runtime
            type(legion_context_f_t), value, intent(in)        :: ctx
            type(legion_logical_region_f_t), value, intent(in) :: handle
        end subroutine
        
        ! @see Legion::LogicalRegion::get_index_space
        function legion_logical_region_get_index_space_f(handle) &
                     bind(C, name="legion_logical_region_get_index_space")
            use iso_c_binding
            import legion_index_space_f_t
            import legion_logical_region_f_t
            implicit none
            
            type(legion_index_space_f_t)                       :: legion_logical_region_get_index_space_f
            type(legion_logical_region_f_t), value, intent(in) :: handle
        end function
        
        ! -----------------------------------------------------------------------
        ! Physical Data Operations
        ! -----------------------------------------------------------------------
        function legion_get_physical_region_by_id_f(regionptr, id, num_regions) &
                     bind(C, name="legion_get_physical_region_by_id")
            use iso_c_binding
            import legion_physical_region_f_t
            implicit none
            
            type(legion_physical_region_f_t)  :: legion_get_physical_region_by_id_f
            type(c_ptr), value, intent(in)    :: regionptr
            integer(c_int), value, intent(in) :: id
            integer(c_int), value, intent(in) :: num_regions
        end function
        
        ! @see Legion::PhysicalRegion::get_field_accessor()
        function legion_physical_region_get_field_accessor_array_1d_f(handle, fid) &
                     bind(C, name="legion_physical_region_get_field_accessor_array_1d")
            use iso_c_binding
            import legion_accessor_array_1d_f_t
            import legion_physical_region_f_t
            implicit none
            
            type(legion_accessor_array_1d_f_t)                  :: legion_physical_region_get_field_accessor_array_1d_f
            type(legion_physical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                   :: fid
        end function
        
        ! @see Legion::PhysicalRegion::get_field_accessor()
        function legion_physical_region_get_field_accessor_array_2d_f(handle, fid) &
                     bind(C, name="legion_physical_region_get_field_accessor_array_2d")
            use iso_c_binding
            import legion_accessor_array_2d_f_t
            import legion_physical_region_f_t
            implicit none
            
            type(legion_accessor_array_2d_f_t)                  :: legion_physical_region_get_field_accessor_array_2d_f
            type(legion_physical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                   :: fid
        end function
        
        ! @see Legion::PhysicalRegion::get_field_accessor()
        function legion_physical_region_get_field_accessor_array_3d_f(handle, fid) &
                     bind(C, name="legion_physical_region_get_field_accessor_array_3d")
            use iso_c_binding
            import legion_accessor_array_3d_f_t
            import legion_physical_region_f_t
            implicit none
            
            type(legion_accessor_array_3d_f_t)                  :: legion_physical_region_get_field_accessor_array_3d_f
            type(legion_physical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                   :: fid
        end function
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        function legion_accessor_array_1d_raw_rect_ptr_f(handle, rect, subrect, offset) &
                     bind(C, name="legion_accessor_array_1d_raw_rect_ptr")
            use iso_c_binding
            import legion_accessor_array_1d_f_t
            import legion_rect_1d_f_t
            import legion_byte_offset_f_t
            implicit none
            
            type(c_ptr)         :: legion_accessor_array_1d_raw_rect_ptr_f
            type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
            type(legion_rect_1d_f_t), value, intent(in)           :: rect
            type(legion_rect_1d_f_t), intent(out)                 :: subrect ! pass reference
            type(legion_byte_offset_f_t), intent(out)             :: offset  ! pass reference
        end function
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        function legion_accessor_array_2d_raw_rect_ptr_f(handle, rect, subrect, offset) &
                     bind(C, name="legion_accessor_array_2d_raw_rect_ptr")
            use iso_c_binding
            import legion_accessor_array_2d_f_t
            import legion_rect_2d_f_t
            import legion_byte_offset_f_t
            implicit none
            
            type(c_ptr)         :: legion_accessor_array_2d_raw_rect_ptr_f
            type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
            type(legion_rect_2d_f_t), value, intent(in)           :: rect
            type(legion_rect_2d_f_t), intent(out)                 :: subrect ! pass reference
            type(legion_byte_offset_f_t), intent(out)             :: offset  ! pass reference
        end function
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        function legion_accessor_array_3d_raw_rect_ptr_f(handle, rect, subrect, offset) &
                     bind(C, name="legion_accessor_array_3d_raw_rect_ptr")
            use iso_c_binding
            import legion_accessor_array_3d_f_t
            import legion_rect_3d_f_t
            import legion_byte_offset_f_t
            implicit none
            
            type(c_ptr)         :: legion_accessor_array_3d_raw_rect_ptr_f
            type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
            type(legion_rect_3d_f_t), value, intent(in)           :: rect
            type(legion_rect_3d_f_t), intent(out)                 :: subrect ! pass reference
            type(legion_byte_offset_f_t), intent(out)             :: offset  ! pass reference
        end function
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_1d_read_point_f(handle, point, dst, bytes) &
                       bind(C, name="legion_accessor_array_1d_read_point")
            use iso_c_binding
            import legion_accessor_array_1d_f_t
            import legion_point_1d_f_t
            implicit none
            
            type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
            type(legion_point_1d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine 
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_2d_read_point_f(handle, point, dst, bytes) &
                       bind(C, name="legion_accessor_array_2d_read_point")
            use iso_c_binding
            import legion_accessor_array_2d_f_t
            import legion_point_2d_f_t
            implicit none
            
            type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
            type(legion_point_2d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine 
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_3d_read_point_f(handle, point, dst, bytes) &
                       bind(C, name="legion_accessor_array_3d_read_point")
            use iso_c_binding
            import legion_accessor_array_3d_f_t
            import legion_point_3d_f_t
            implicit none
            
            type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
            type(legion_point_3d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine 
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_1d_write_point_f(handle, point, src, bytes) &
                       bind(C, name="legion_accessor_array_1d_write_point")
            use iso_c_binding
            import legion_accessor_array_1d_f_t
            import legion_point_1d_f_t
            implicit none
            
            type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
            type(legion_point_1d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: src
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine 
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_2d_write_point_f(handle, point, src, bytes) &
                       bind(C, name="legion_accessor_array_2d_write_point")
            use iso_c_binding
            import legion_accessor_array_2d_f_t
            import legion_point_2d_f_t
            implicit none
            
            type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
            type(legion_point_2d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: src
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine 
        
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_3d_write_point_f(handle, point, src, bytes) &
                       bind(C, name="legion_accessor_array_3d_write_point")
            use iso_c_binding
            import legion_accessor_array_3d_f_t
            import legion_point_3d_f_t
            implicit none
            
            type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
            type(legion_point_3d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: src
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine 
        
        ! -----------------------------------------------------------------------
        ! Combined Operations
        ! -----------------------------------------------------------------------
        function legion_task_get_index_space_from_logical_region_f(handle, tid) &
            bind (C, name="legion_task_get_index_space_from_logical_region")
            use iso_c_binding
            import legion_index_space_f_t
            import legion_task_f_t
            implicit none
            
            type(legion_index_space_f_t)             :: legion_task_get_index_space_from_logical_region_f
            type(legion_task_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)        :: tid
        end function
    end interface
end module legion_fortran