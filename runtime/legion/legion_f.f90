#define NEW_OPAQUE_TYPE_F(T) type, bind(C) :: T; type(c_ptr) :: impl; end type T
module legion_fortran
    use, intrinsic :: iso_c_binding
    ! C NEW_OPAQUE_TYPE_F
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
            
            type(legion_future_f_t)                     :: legion_task_launcher_execute_f
            type(legion_runtime_f_t), value, intent(in)        :: runtime
            type(legion_context_f_t), value, intent(in)        :: ctx
            type(legion_task_launcher_f_t), value, intent(in)  :: launcher
        end function
        
        function legion_predicate_true_f() bind(C, name="legion_predicate_true")
            use iso_c_binding
            import legion_predicate_f_t
            implicit none
            
            type(legion_predicate_f_t)  :: legion_predicate_true_f
        end function
        
        function legion_task_get_args_f(task) bind(C, name="legion_task_get_args")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            type(c_ptr)                         :: legion_task_get_args_f
            type(legion_task_f_t), value, intent(in)   :: task
        end function
        
        function legion_task_get_arglen_f(task) bind(C, name="legion_task_get_arglen")
            use iso_c_binding
            import legion_task_f_t
            implicit none
            
            integer(c_size_t)                   :: legion_task_get_arglen_f
            type(legion_task_f_t), value, intent(in)   :: task
        end function
    
    end interface
end module legion_fortran