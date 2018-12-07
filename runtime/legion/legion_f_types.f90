module legion_fortran_types
    use, intrinsic :: iso_c_binding
    implicit none
    
    ! legion_privilege_mode_t
    integer(c_int), parameter :: NO_ACCESS = Z'00000000'
    integer(c_int), parameter :: READ_PRIV = Z'00000001'
    integer(c_int), parameter :: READ_ONLY = Z'00000001'
    integer(c_int), parameter :: WRITE_PRIV = Z'00000002'
    integer(c_int), parameter :: REDUCE_PRIV = Z'00000004'
    integer(c_int), parameter :: REDUCE = Z'00000004'
    integer(c_int), parameter :: READ_WRITE = Z'00000007'
    integer(c_int), parameter :: DISCARD_MASK = Z'10000000'
    integer(c_int), parameter :: WRITE_ONLY = Z'10000002'
    integer(c_int), parameter :: WRITE_DISCARD = Z'10000007'
    
    ! legion_coherence_property_t
    integer(c_int), parameter :: EXCLUSIVE = 0
    integer(c_int), parameter :: ATOMIC = 1
    integer(c_int), parameter :: SIMULTANEOUS = 2
    integer(c_int), parameter :: RELAXED = 3
    
    !legion_file_mode_t
    integer(c_int), parameter :: LEGION_FILE_READ_ONLY = 0
    integer(c_int), parameter :: LEGION_FILE_READ_WRITE = 0
    integer(c_int), parameter :: LEGION_FILE_CREATE = 0
    
    !legion_processor_kind_t
    integer(c_int), parameter :: NO_KIND = 0
    integer(c_int), parameter :: TOC_PROC = 1
    integer(c_int), parameter :: LOC_PROC = 2
    integer(c_int), parameter :: UTIL_PROC = 3
    integer(c_int), parameter :: IO_PROC = 4
    integer(c_int), parameter :: PROC_GROUP = 5
    integer(c_int), parameter :: PROC_SET = 6
    integer(c_int), parameter :: OMP_PROC = 7
    integer(c_int), parameter :: PY_PROC = 8
    
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
end module 