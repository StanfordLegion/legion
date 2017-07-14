module legion_fortran
    use, intrinsic :: iso_c_binding
    
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
    NEW_OPAQUE_TYPE_F(legion_accessor_generic_f_t)
    NEW_OPAQUE_TYPE_F(legion_accessor_array_f_t)
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

    ! legion domain
    type, bind(C) :: legion_domain_f_t
        integer(c_long_long)                                  :: is_id
        integer(c_int)                                        :: dim
        ! check MAX_DOMAIN_DIM = 2 * REALM_MAX_RECT_DIM
#define MAX_DOMAIN_DIM_F 6 
        integer(c_long_long), dimension(0:MAX_DOMAIN_DIM_F-1) :: rect_data
#undef MAX_DOMAIN_DIM_F        
    end type legion_domain_f_t
    
    ! domain point
    type, bind(C) :: legion_domain_point_f_t
        integer(c_int)                                        :: dim
#define MAX_POINT_DIM_F 6
        integer(c_long_long), dimension(0:MAX_POINT_DIM_F-1) :: point_data
#undef  MAX_POINT_DIM_F
    end type legion_domain_point_f_t
    
    type, bind(C) :: legion_task_config_options_f_t
        logical(c_bool) :: leaf
        logical(c_bool) :: inner
        logical(c_bool) :: idempotent
    end type legion_task_config_options_f_t
    
    type, bind(C) :: legion_task_argument_f_t
        type(c_ptr)         :: args
        integer(c_size_t)   :: arglen
    end type legion_task_argument_f_t
    
    ! C typedef enum
  !  enum, bind(C) :: legion_processor_kind_t
   !     enumrator :: NO_KIND = 0
    !    TOC_PROC, LOC_PROC, UTIL_PROC, IO_PROC, PROC_GROUP, PROC_SET, OMP_PROC
    !end enum

    interface
        subroutine legion_runtime_set_top_level_task_id_f(top_id) bind(C, name="legion_runtime_set_top_level_task_id")
            use iso_c_binding
            implicit none
            
            integer(c_int), value, intent(in) :: top_id
        end subroutine legion_runtime_set_top_level_task_id_f
        
        function legion_execution_constraint_set_create_f() bind(C, name="legion_execution_constraint_set_create")
            use iso_c_binding
            import legion_execution_constraint_set_f_t
            implicit none
            
            type(legion_execution_constraint_set_f_t) :: legion_execution_constraint_set_create_f
        end function
        
        subroutine legion_execution_constraint_set_add_processor_constraint_f(handle, proc_kind) &
                   bind(C, name="legion_execution_constraint_set_add_processor_constraint")
            use iso_c_binding
            import legion_execution_constraint_set_f_t
            implicit none
            
            type(legion_execution_constraint_set_f_t), value, intent(in)    :: handle
            integer(c_int), value, intent(in)                               :: proc_kind
        end subroutine
        
        function legion_task_layout_constraint_set_create_f() bind(C, name="legion_task_layout_constraint_set_create")
            use iso_c_binding
            import legion_task_layout_constraint_set_f_t
            implicit none
            
            type(legion_task_layout_constraint_set_f_t) :: legion_task_layout_constraint_set_create_f
        end function
        
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
        
        function legion_runtime_start_f(argc, argv, background) bind(C, name="legion_runtime_start")
            use iso_c_binding
            implicit none
            
            integer(c_int)                      :: legion_runtime_start_f
            integer(c_int), value, intent(in)   :: argc
            type(c_ptr), value, intent(in)      :: argv
            logical(c_bool), value, intent(in)  :: background
        end function
        
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
            type(legion_task_f_t), intent(inout)            :: task ! pass reference
            type(legion_physical_region_f_t), intent(inout) :: regionptr
            integer(c_int), intent(inout)                   :: num_regions ! pass reference
            type(legion_context_f_t), intent(inout)         :: ctx ! pass reference          
            type(legion_runtime_f_t), intent(inout)         :: runtime ! pass reference
        end subroutine
        
        subroutine legion_task_postamble_f(runtime, ctx, retval, retsize) bind(C, name="legion_task_postamble")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            implicit none
            
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
            type(c_ptr), value, intent(in)              :: retval
            integer(c_size_t), value, intent(in)        :: retsize
        end subroutine
        
        ! task launcher
        function legion_task_launcher_create_f(tid, arg, pred, id, tag) bind(C, name="legion_task_launcher_create")
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
        
        function legion_task_launcher_execute_f(runtime, ctx, launcher) bind(C, name="legion_task_launcher_execute")
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
        
        ! index launcher
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
        
        function legion_predicate_true_f() bind(C, name="legion_predicate_true")
            use iso_c_binding
            import legion_predicate_f_t
            implicit none
            
            type(legion_predicate_f_t)  :: legion_predicate_true_f
        end function
        
        ! argument map
        function legion_argument_map_create_f() bind(C, name="legion_argument_map_create")
            use iso_c_binding
            import legion_argument_map_f_t
            
            implicit none
            
            type(legion_argument_map_f_t) :: legion_argument_map_create_f
        end function
        
        subroutine legion_argument_map_set_point_f(map, dp, arg, replace) bind(C, name="legion_argument_map_set_point")
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
        
        ! task args
        function legion_task_get_args_f(task) bind(C, name="legion_task_get_args")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            type(c_ptr)                              :: legion_task_get_args_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        function legion_task_get_arglen_f(task) bind(C, name="legion_task_get_arglen")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            integer(c_size_t)                        :: legion_task_get_arglen_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        function legion_task_get_local_args_f(task) bind(C, name="legion_task_get_local_args")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            type(c_ptr)                              :: legion_task_get_local_args_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        function legion_task_get_local_arglen_f(task) bind(C, name="legion_task_get_local_arglen")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            integer(c_size_t)                        :: legion_task_get_local_arglen_f
            type(legion_task_f_t), value, intent(in) :: task
        end function
        
        ! legion domain
        function legion_domain_from_rect_1d_f(r) bind(C, name="legion_domain_from_rect_1d")
            use iso_c_binding
            import legion_rect_1d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_domain_f_t)                     :: legion_domain_from_rect_1d_f
            type(legion_rect_1d_f_t), value, intent(in) :: r
        end function
        
        function legion_domain_from_rect_2d_f(r) bind(C, name="legion_domain_from_rect_2d")
            use iso_c_binding
            import legion_rect_2d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_domain_f_t)                     :: legion_domain_from_rect_2d_f
            type(legion_rect_2d_f_t), value, intent(in) :: r
        end function
        
        function legion_domain_from_rect_3d_f(r) bind(C, name="legion_domain_from_rect_3d")
            use iso_c_binding
            import legion_rect_3d_f_t
            import legion_domain_f_t
            implicit none
            
            type(legion_domain_f_t)                     :: legion_domain_from_rect_3d_f
            type(legion_rect_3d_f_t), value, intent(in) :: r
        end function
        
        function legion_domain_point_from_point_1d_f(p) bind(C, name="legion_domain_point_from_point_1d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_1d_f_t
            implicit none
            
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_1d_f
            type(legion_point_1d_f_t), value, intent(in) :: p
        end function
        
        function legion_domain_point_from_point_2d_f(p) bind(C, name="legion_domain_point_from_point_2d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_2d_f_t
            implicit none
            
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_2d_f
            type(legion_point_2d_f_t), value, intent(in) :: p
        end function
        
        function legion_domain_point_from_point_3d_f(p) bind(C, name="legion_domain_point_from_point_3d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_3d_f_t
            implicit none
            
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_3d_f
            type(legion_point_3d_f_t), value, intent(in) :: p
        end function
        
        ! future map
        subroutine legion_future_map_wait_all_results_f(handle) bind(C, name="legion_future_map_wait_all_results")
            use iso_c_binding
            import legion_future_map_f_t
            implicit none
            
            type(legion_future_map_f_t), value, intent(in) :: handle
        end subroutine
    
    end interface
end module legion_fortran