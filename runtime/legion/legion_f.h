! ===============================================================================
! Predicate
! ===============================================================================
type FPredicate
  type(legion_predicate_f_t) :: predicate
end type FPredicate

interface FPredicate
  module procedure legion_predicate_constructor
end interface FPredicate
! ===============================================================================
! DomainPoint
! ===============================================================================
type FDomainPoint
  type(legion_domain_point_f_t) :: point
contains
  procedure :: get_point_1d => legion_domain_point_get_point_1d
  procedure :: get_point_2d => legion_domain_point_get_point_2d
  procedure :: get_point_3d => legion_domain_point_get_point_3d
end type FDomainPoint

interface FDomainPoint
  module procedure legion_domain_point_constructor_point_1d
  module procedure legion_domain_point_constructor_point_2d
  module procedure legion_domain_point_constructor_point_3d
end interface FDomainPoint

! ===============================================================================
! Point
! ===============================================================================
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
end interface FPoint1D

interface FPoint2D
  module procedure legion_point_2d_constructor_integer4
  module procedure legion_point_2d_constructor_integer8
end interface FPoint2D

interface FPoint3D
  module procedure legion_point_3d_constructor_integer4
  module procedure legion_point_3d_constructor_integer8
end interface FPoint3D

! ===============================================================================
! Domain
! ===============================================================================
type FDomain
  type(legion_domain_f_t) :: domain
contains
  procedure :: get_volume => legion_domain_get_volume
end type FDomain

! ===============================================================================
! Rect
! ===============================================================================
type FRect1D
  type(legion_rect_1d_f_t) :: rect
contains
  procedure :: get_volume => legion_rect_1d_get_volume
end type FRect1D

type FRect2D
  type(legion_rect_2d_f_t) :: rect
contains
  procedure :: get_volume => legion_rect_2d_get_volume
end type FRect2D

type FRect3D
  type(legion_rect_3d_f_t) :: rect
contains
  procedure :: get_volume => legion_rect_3d_get_volume
end type FRect3D

interface FRect1D
  module procedure legion_rect_1d_constructor_integer4
  module procedure legion_rect_1d_constructor_integer8
  module procedure legion_rect_1d_constructor_point_1d
end interface FRect1D

interface FRect2D
  module procedure legion_rect_2d_constructor_point_2d
end interface FRect2D

interface FRect3D
  module procedure legion_rect_3d_constructor_point_3d
end interface FRect3D

interface assignment (=)
  module procedure legion_rect_1d_assignment_from_domain
  module procedure legion_rect_2d_assignment_from_domain
  module procedure legion_rect_3d_assignment_from_domain    
end interface

! ===============================================================================
! DomainTransform
! ===============================================================================
type FDomainTransform
  type(legion_domain_transform_f_t) :: transform
end type FDomainTransform

! ===============================================================================
! Transform
! ===============================================================================
type FTransform1X1
  type(legion_transform_1x1_f_t) :: transform
end type FTransform1X1

type FTransform2X2
  type(legion_transform_2x2_f_t) :: transform
end type FTransform2X2

type FTransform3X3
  type(legion_transform_3x3_f_t) :: transform
end type FTransform3X3

! ===============================================================================
! FieldAccessor
! ===============================================================================
type FFieldAccessor
  integer :: dim
  integer(c_size_t) :: data_size
  integer(c_int) :: privilege_mode
contains
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
contains
  procedure :: get_raw_ptr => legion_field_accessor_1d_get_raw_ptr
end type FFieldAccessor1D

type, extends(FFieldAccessor) :: FFieldAccessor2D
  type(legion_accessor_array_2d_f_t) :: accessor
contains
  procedure :: get_raw_ptr => legion_field_accessor_2d_get_raw_ptr
end type FFieldAccessor2D

type, extends(FFieldAccessor) :: FFieldAccessor3D
  type(legion_accessor_array_3d_f_t) :: accessor
contains
  procedure :: get_raw_ptr => legion_field_accessor_3d_get_raw_ptr
end type FFieldAccessor3D

interface FFieldAccessor1D
  module procedure legion_field_accessor_1d_constructor
end interface FFieldAccessor1D

interface FFieldAccessor2D
  module procedure legion_field_accessor_2d_constructor
end interface FFieldAccessor2D

interface FFieldAccessor3D
  module procedure legion_field_accessor_3d_constructor
end interface FFieldAccessor3D

! ===============================================================================
! FDomainPointIterator
! ===============================================================================
type FDomainPointIterator
  type(legion_domain_point_iterator_f_t) :: iterator    
contains
  ! @see Legion::Domain::DomainPointIterator::step()
  procedure :: step => legion_domain_point_iterator_step
  
  ! @see Legion::Domain::DomainPointIterator::any_left
  procedure :: has_next => legion_domain_point_iterator_has_next
  
  ! @see Legion::Domain::DomainPointIterator::~DomainPointIterator()
  procedure :: destroy => legion_domain_point_iterator_destructor
    
end type FDomainPointIterator

interface FDomainPointIterator
  ! @see Legion::Domain::DomainPointIterator::DomainPointIterator()
  module procedure legion_domain_point_iterator_constructor_from_domain
  module procedure legion_domain_point_iterator_constructor_from_rect_1d
  module procedure legion_domain_point_iterator_constructor_from_rect_2d
  module procedure legion_domain_point_iterator_constructor_from_rect_3d
end interface FDomainPointIterator

! ===============================================================================
! IndexSpace
! ===============================================================================
type FIndexSpace
  type(legion_index_space_f_t) :: is
end type FIndexSpace

! ===============================================================================
! FieldSpace
! ===============================================================================
type FFieldSpace
  type(legion_field_space_f_t) :: fs
end type FFieldSpace

! ===============================================================================
! FieldAllocator
! ===============================================================================
type FFieldAllocator
  type(legion_field_allocator_f_t) :: fa
contains
  ! @see Legion::FieldAllocator::allocate_field()
  procedure :: allocate_field => legion_field_allocator_allocate_field
  
  ! @see Legion::FieldAllocator::free_field()
  procedure :: free_field => legion_field_allocator_free_field
  
  ! @see Legion::FieldAllocator::~FieldAllocator()
  procedure :: destroy => legion_field_allocator_destroy
end type FFieldAllocator

! ===============================================================================
! IndexPartition
! ===============================================================================
type FIndexPartition
  type(legion_index_partition_f_t) :: ip
end type FIndexPartition

! ===============================================================================
! LogicalRegion
! ===============================================================================
type FLogicalRegion
  type(legion_logical_region_f_t) :: lr
contains
  ! @see Legion::LogicalRegion::get_index_space
  procedure :: get_index_space => legion_logical_region_get_index_space
end type FLogicalRegion

! ===============================================================================
! LogicalPartition
! ===============================================================================
type FLogicalPartition
  type(legion_logical_partition_f_t) :: lp
end type FLogicalPartition

! ===============================================================================
! PhysicalRegion
! ===============================================================================
type FPhysicalRegion
  type(legion_physical_region_f_t) :: pr
contains
  ! @see Legion::PhysicalRegion::is_mapped()
  procedure :: is_mapped => legion_physical_region_is_mapped
  
  ! @see Legion::PhysicalRegion::wait_until_valid()
  procedure :: wait_until_valid => legion_physical_region_wait_until_valid
  
  ! @see Legion::PhysicalRegion::is_valid()
  procedure :: is_valid => legion_physical_region_is_valid
  
  ! @see Legion::PhysicalRegion::~PhysicalRegion()
  procedure :: destroy =>legion_physical_region_destructor
end type FPhysicalRegion

! ===============================================================================
! PhysicalRegionList
! ===============================================================================
type FPhysicalRegionList
  type(c_ptr) :: region_ptr
  integer :: num_regions
contains
  procedure :: size => legion_physical_region_list_size
  procedure :: get_region => legion_physical_region_list_get_region_by_id
end type FPhysicalRegionList

! ===============================================================================
! RegionRequirement
! ===============================================================================
type FRegionRequirement
  type(legion_region_requirement_f_t) :: rr
contains
  ! @see Legion::RegionRequirement::region
  procedure :: get_region => legion_region_requirement_get_logical_region
  
  ! @see Legion::RegionRequirement::privilege_fields
  procedure :: get_privilege_field => legion_region_requirement_get_privilege_field_by_id
end type FRegionRequirement

! ===============================================================================
! Future
! ===============================================================================
type FFuture
  type(legion_future_f_t) :: future
contains
  procedure, private :: legion_future_get_integer4
  procedure, private :: legion_future_get_integer8
  procedure, private :: legion_future_get_real4
  procedure, private :: legion_future_get_real8
  procedure, private :: legion_future_get_untyped
  ! @see Legion::Future::get_result
  generic :: get_result => legion_future_get_integer4, &
                           legion_future_get_integer8, &
                           legion_future_get_real4, &
                           legion_future_get_real8, &
                           legion_future_get_untyped
  
  ! @see Legion::Future::~Future()
  procedure :: destroy => legion_future_destructor
end type FFuture

interface FFuture
  ! @see Legion::Future::Future()
  module procedure :: legion_future_constructor
end interface FFuture

! ===============================================================================
! FutureMap
! ===============================================================================
type FFutureMap
  type(legion_future_map_f_t) :: future_map
contains
  procedure, private :: legion_future_map_get_future_index
  procedure, private :: legion_future_map_get_future_domain_point
  procedure, private :: legion_future_map_get_future_point_1d
  procedure, private :: legion_future_map_get_future_point_2d
  procedure, private :: legion_future_map_get_future_point_3d
  
  ! @see Legion::Future::get_future()
  generic :: get_future => legion_future_map_get_future_index
  
  ! @see Legion::FutureMap::wait_all_results()
  procedure :: wait_all_results => legion_future_map_wait_all_results
end type FFutureMap

! ===============================================================================
! TaskArgument
! ===============================================================================
type FTaskArgument
  type(legion_task_argument_f_t) :: task_arg
end type FTaskArgument

interface FTaskArgument
  module procedure legion_task_argument_constructor
end interface FTaskArgument

! ===============================================================================
! ArgumentMap
! ===============================================================================
type FArgumentMap
  type(legion_argument_map_f_t) :: arg_map
contains
  procedure, private :: legion_argument_map_set_point_domain_point
  procedure, private :: legion_argument_map_set_point_integer
  procedure, private :: legion_argument_map_set_point_1d_point
  procedure, private :: legion_argument_map_set_point_2d_point
  procedure, private :: legion_argument_map_set_point_3d_point
  
  ! @see Legion::ArgumentMap::set_point()
  generic :: set_point => legion_argument_map_set_point_domain_point, &
                          legion_argument_map_set_point_integer, &
                          legion_argument_map_set_point_1d_point, &
                          legion_argument_map_set_point_2d_point, &
                          legion_argument_map_set_point_3d_point
  
  ! @see Legion::ArgumentMap::~ArgumentMap()
  procedure :: destroy => legion_argument_map_destructor  
end type FArgumentMap

interface FArgumentMap
  module procedure legion_argument_map_constructor
end interface FArgumentMap

! ===============================================================================
! TaskLauncher
! ===============================================================================
type FTaskLauncher
  type(legion_task_launcher_f_t) :: launcher
contains
  procedure, private :: legion_task_launcher_add_region_requirement
  
  ! @see Legion::TaskLauncher::add_region_requirement()  
  generic :: add_region_requirement => legion_task_launcher_add_region_requirement
  
  ! @see Legion::TaskLauncher::add_field()
  procedure :: add_field => legion_task_launcher_add_field
  
  ! @see Legion::TaskLauncher::add_future()
  procedure :: add_future => legion_task_launcher_add_future
  
  ! Legion::TaskLauncher::~TaskLauncher()
  procedure :: destroy => legion_task_launcher_destructor
end type FTaskLauncher

interface FTaskLauncher
  module procedure legion_task_launcher_constructor
end interface FTaskLauncher

! ===============================================================================
! IndexLauncher
! ===============================================================================
type FIndexLauncher
  type(legion_index_launcher_f_t) :: index_launcher
contains
  procedure, private :: legion_index_launcher_add_region_requirement_logical_partition
  
  ! @see Legion::IndexTaskLauncher::add_region_requirement()  
  generic :: add_region_requirement => legion_index_launcher_add_region_requirement_logical_partition
  
  ! @see Legion::IndexLaunchxer::add_field()
  procedure :: add_field => legion_index_launcher_add_field
  
  ! @see Legion::IndexTaskLauncher::~IndexTaskLauncher()
  procedure :: destroy => legion_index_launcher_destructor
end type FIndexLauncher

interface FIndexLauncher
  module procedure legion_index_launcher_constructor_from_index_space
end interface FIndexLauncher

! ===============================================================================
! InlineLauncher
! ===============================================================================
type FInlineLauncher
  type(legion_inline_launcher_f_t) :: inline_launcher
contains
  ! @see Legion::InlineLauncher::add_field()
  procedure :: add_field => legion_inline_launcher_add_field
  
  !
  procedure :: destroy => legion_inline_launcher_destructor
end type FInlineLauncher

interface FInlineLauncher
  ! @see Legion::InlineLauncher::InlineLauncher()
  module procedure legion_inline_launcher_constructor
end interface FInlineLauncher

! ===============================================================================
! CopyLauncher
! ===============================================================================
type FCopyLauncher
  type(legion_copy_launcher_f_t) :: copy_launcher
contains
  ! @see Legion::CopyLauncher::add_copy_requirements()
  procedure :: add_src_requirements => legion_copy_launcher_add_src_requirements
  
  ! @see Legion::CopyLauncher::add_copy_requirements()
  procedure :: add_dst_requirements => legion_copy_launcher_add_dst_requirements
  
  ! @see Legion::CopyLauncher::add_src_field()
  procedure :: add_src_field => legion_copy_launcher_add_src_field
  
  ! @see Legion::CopyLauncher::add_dst_field()
  procedure :: add_dst_field => legion_copy_launcher_add_dst_field
  
  ! @see Legion::CopyLauncher::~CopyLauncher()
  procedure :: destroy => legion_copy_launcher_destructor  
end type FCopyLauncher

interface FCopyLauncher
  ! @see Legion::CopyLauncher::CopyLauncher()
  module procedure legion_copy_launcher_constructor
end interface FCopyLauncher

! ===============================================================================
! IndexCopyLauncher
! ===============================================================================
type FIndexCopyLauncher
  type(legion_index_copy_launcher_f_t) :: index_copy_launcher
contains
  procedure, private :: legion_index_copy_launcher_add_src_requirements_lr
  procedure, private :: legion_index_copy_launcher_add_src_requirements_lp
  
  ! @see Legion::IndexCopyLauncher::add_copy_requirements()
  generic :: add_src_requirements => legion_index_copy_launcher_add_src_requirements_lr, &
                                     legion_index_copy_launcher_add_src_requirements_lp
  
  procedure, private :: legion_index_copy_launcher_add_dst_requirements_lr
  procedure, private :: legion_index_copy_launcher_add_dst_requirements_lp
  
  ! @see Legion::IndexCopyLauncher::add_copy_requirements()
  generic :: add_dst_requirements => legion_index_copy_launcher_add_dst_requirements_lr, &
                                     legion_index_copy_launcher_add_dst_requirements_lp
  
  ! @see Legion::IndexCopyLauncher::add_src_field()
  procedure :: add_src_field => legion_index_copy_launcher_add_src_field
  
  ! @see Legion::IndexCopyLauncher::add_dst_field()
  procedure :: add_dst_field => legion_index_copy_launcher_add_dst_field
  
  ! @see Legion::IndexCopyLauncher::~IndexCopyLauncher()
  procedure :: destroy => legion_index_copy_launcher_destructor  
end type FIndexCopyLauncher

interface FIndexCopyLauncher
  ! @see Legion::IndexCopyLauncher::IndexCopyLauncher()
  module procedure legion_index_copy_launcher_constructor
end interface FIndexCopyLauncher

! ===============================================================================
! AttachLauncher
! ===============================================================================
type FAttachLauncher
  type(legion_attach_launcher_f_t) :: attach_launcher
contains  
  ! @see Legion::AttachLauncher::attach_array_soa()
  procedure :: attach_array_soa => legion_attach_launcher_attach_array_soa
  
  ! @see Legion::AttachLauncher::~AttachLauncher()
  procedure :: destroy => legion_attach_launcher_destructor
end type FAttachLauncher

interface FAttachLauncher
  ! @see Legion::AttachLauncher::AttachLauncher()
  module procedure legion_attach_launcher_constructor
end interface FAttachLauncher

! ===============================================================================
! Task
! ===============================================================================
type FTask
  type(legion_task_f_t) :: task
contains
  ! @see Legion::Task::regions
  procedure :: get_region_requirement => legion_task_get_region_requirement_by_id
  
  ! @see Legion::Task::arglen
  procedure :: get_arglen=> legion_task_get_arglen
  
  ! @see Legion::Task::args
  procedure :: get_args => legion_task_get_args
  
  ! @see Legion::Task::local_arglen
  procedure :: get_local_arglen => legion_task_get_local_arglen
  
  ! @see Legion::Task::local_args
  procedure :: get_local_args => legion_task_get_local_args
  
  ! @see Legion::Task::futures
  procedure :: get_future => legion_task_get_future
  
  ! @see Legion::Task::futures
  procedure :: get_futures_size => legion_task_get_futures_size
end type FTask

! ===============================================================================
! Runtime
! ===============================================================================
type FRuntime
  type(legion_runtime_f_t) :: runtime
contains
  procedure, private :: legion_runtime_create_index_space_from_elmts_size
  procedure, private :: legion_runtime_create_index_space_from_domain
  procedure, private :: legion_runtime_create_index_space_from_rect_1d
  procedure, private :: legion_runtime_create_index_space_from_rect_2d
  procedure, private :: legion_runtime_create_index_space_from_rect_3d
  
  ! @see Legion::Runtime::get_index_space_domain()
  procedure :: get_index_space_domain => legion_runtime_get_index_domain_return_domain
                                       
  ! @see Legion::Runtime::create_index_space()
  generic :: create_index_space => legion_runtime_create_index_space_from_elmts_size, &
                                   legion_runtime_create_index_space_from_domain, &
                                   legion_runtime_create_index_space_from_rect_1d, &
                                   legion_runtime_create_index_space_from_rect_2d, &
                                   legion_runtime_create_index_space_from_rect_3d
  
  ! @see Legion::Runtime::destroy_index_space()
  procedure :: destroy_index_space => legion_runtime_destroy_index_space
  
  ! @see Legion::Runtime::create_field_space()
  procedure :: create_field_space => legion_runtime_create_field_space 
  
  ! @see Legion::Runtime::destroy_field_space()
  procedure :: destroy_field_space => legion_runtime_destroy_field_space
  
  ! @see Legion::Runtime::create_field_allocator()
  procedure :: create_field_allocator => legion_runtime_create_field_allocator
  
  ! @see Legion::Runtime::create_logical_region()
  procedure :: create_logical_region => legion_runtime_create_logical_region
  
  ! @see Legion::Runtime::destroy_logical_region()
  procedure :: destroy_logical_region => legion_runtime_destroy_logical_region
  
  ! @see Legion::Runtime::create_equal_partition()
  procedure :: create_equal_partition => legion_runtime_create_equal_partition
  
  procedure, private :: legion_runtime_create_partition_by_restriction_domain_transform
  procedure, private :: legion_runtime_create_partition_by_restriction_transform_1x1
  procedure, private :: legion_runtime_create_partition_by_restriction_transform_2x2
  procedure, private :: legion_runtime_create_partition_by_restriction_transform_3x3
  
  ! @see Legion::Runtime::create_partition_by_restriction()
  generic :: create_partition_by_restriction => &
    legion_runtime_create_partition_by_restriction_domain_transform, &
    legion_runtime_create_partition_by_restriction_transform_1x1, &
    legion_runtime_create_partition_by_restriction_transform_2x2, &
    legion_runtime_create_partition_by_restriction_transform_3x3                                            
  
  ! @see Legion::Runtime::get_logical_partition()
  procedure :: get_logical_partition => legion_runtime_get_logical_partition
  
  ! @see Legion::Runtime::execute_task()
  procedure :: execute_task => legion_runtime_execute_task
  
  ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &)
  procedure :: execute_index_space => legion_runtime_execute_index_space
  
  ! @see Legion::Runtime::map_region()
  procedure :: map_region => legion_runtime_map_region
  
  ! @see Legion::Runtime::unmap_region()
  procedure :: unmap_region => legion_runtime_unmap_region
  
  ! @see Legion::Runtime::remap_region()
  procedure :: remap_region => legion_runtime_remap_region
  
  procedure, private :: legion_runtime_fill_field_integer4
  procedure, private :: legion_runtime_fill_field_integer8
  procedure, private :: legion_runtime_fill_field_real4
  procedure, private :: legion_runtime_fill_field_real8
  procedure, private :: legion_runtime_fill_field_ptr
  
  ! @see Legion::Runtime::fill_field()
  generic :: fill_field => legion_runtime_fill_field_integer4, &
                           legion_runtime_fill_field_integer8, &
                           legion_runtime_fill_field_real4, &
                           legion_runtime_fill_field_real8, &
                           legion_runtime_fill_field_ptr
                           
  procedure, private :: legion_runtime_issue_copy_operation_single
  procedure, private :: legion_runtime_issue_copy_operation_index
  
  ! @see Legion::Runtime::issue_copy_operation()
  generic :: issue_copy_operation => legion_runtime_issue_copy_operation_single, &
                                     legion_runtime_issue_copy_operation_index

  ! @see Legion::Runtime::attach_external_resource()
  procedure :: attach_external_resource => legion_runtime_attach_external_resource  
  
  procedure, private :: legion_runtime_detach_external_resource
  procedure, private :: legion_runtime_detach_external_resource_flush
  procedure, private :: legion_runtime_detach_external_resource_unordered

  ! @see Legion::Runtime::detach_external_resource()
  generic :: detach_external_resource => legion_runtime_detach_external_resource, &
                                         legion_runtime_detach_external_resource_flush, &
                                         legion_runtime_detach_external_resource_unordered
end type FRuntime

! ===============================================================================
! Context
! ===============================================================================
type FContext
  type(legion_context_f_t) :: context
end type FContext

! ===============================================================================
! ProcessorConstraint
! ===============================================================================
type FProcessorConstraint
  integer(c_int) :: proc_kind
end type FProcessorConstraint

interface FProcessorConstraint
  module procedure legion_processor_constraint_constructor
end interface FProcessorConstraint

! ===============================================================================
! TaskVariantRegistrar
! ===============================================================================
type FTaskVariantRegistrar
  type(legion_execution_constraint_set_f_t) :: execution_constraints
  type(legion_task_layout_constraint_set_f_t) :: task_layout_constraints
  type(legion_task_config_options_f_t) :: config_options
  integer(c_int) :: task_id
contains
  ! @see Legion::ExecutionConstraintSet::add_constraint(Legion::ProcessorConstraint)
  procedure, private :: legion_task_variant_registrar_add_processor_constraint
  
  generic :: add_constraint => legion_task_variant_registrar_add_processor_constraint
  
  ! @see Legion::ExecutionConstraintSet::~ExecutionConstraintSet()
  ! @see Legion::TaskLayoutConstraintSet::~TaskLayoutConstraintSet()
  procedure :: destroy => legion_task_variant_registrar_destructor
  
  ! @see Legion::TaskVariantRegistrar::set_leaf
  procedure :: set_leaf => legion_task_variant_registrar_set_leaf
  
  ! @see Legion::TaskVariantRegistrar::set_inner
  procedure :: set_inner => legion_task_variant_registrar_set_inner
  
  ! @see Legion::TaskVariantRegistrar::set_idempotent
  procedure :: set_idempotent => legion_task_variant_registrar_set_idempotent
  
  ! @see Legion::TaskVariantRegistrar::set_replicable
  procedure :: set_replicable => legion_task_variant_registrar_set_replicable
end type FTaskVariantRegistrar

interface FTaskVariantRegistrar
  module procedure legion_task_variant_registrar_constructor
end interface FTaskVariantRegistrar