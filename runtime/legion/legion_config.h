/* Copyright 2017 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __LEGION_CONFIG_H__
#define __LEGION_CONFIG_H__

// for UINT_MAX, INT_MAX, INT_MIN
#include <limits.h>

/**
 * \file legion_config.h
 */

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++. Keep any C++-isms in
// legion_types.h, or elsewhere.
//
// ******************** IMPORTANT **************************

#include "lowlevel_config.h"

//==========================================================================
//                                Constants
//==========================================================================

#define AUTO_GENERATE_ID   UINT_MAX

#define GC_MIN_PRIORITY    INT_MIN
#define GC_MAX_PRIORITY    INT_MAX

#define GC_FIRST_PRIORITY  GC_MAX_PRIORITY
#define GC_DEFAULT_PRIORITY 0
#define GC_LAST_PRIORITY   (GC_MIN_PRIORITY+1)
#define GC_NEVER_PRIORITY  GC_MIN_PRIORITY

#ifndef MAX_RETURN_SIZE
#define MAX_RETURN_SIZE    2048 // maximum return type size in bytes
#endif

#ifndef MAX_FIELDS
#define MAX_FIELDS         512 // must be a power of 2
#endif

// Some default values

// The maximum number of nodes to be run on
#ifndef MAX_NUM_NODES
#define MAX_NUM_NODES                   1024
#endif
// The maximum number of processors on a node
#ifndef MAX_NUM_PROCS
#define MAX_NUM_PROCS                   64
#endif
// Maximum ID for an application task ID 
#ifndef MAX_APPLICATION_TASK_ID
#define MAX_APPLICATION_TASK_ID         (1<<20)
#endif
// Maximum ID for an application field ID
#ifndef MAX_APPLICATION_FIELD_ID
#define MAX_APPLICATION_FIELD_ID        (1<<20)
#endif
// Maximum ID for an application mapper ID
#ifndef MAX_APPLICATION_MAPPER_ID
#define MAX_APPLICATION_MAPPER_ID       (1<<20)
#endif
// Maximum ID for an application projection ID
#ifndef MAX_APPLICATION_PROJECTION_ID
#define MAX_APPLICATION_PROJECTION_ID   (1<<20)
#endif
// Default number of local fields per field space
#ifndef DEFAULT_LOCAL_FIELDS
#define DEFAULT_LOCAL_FIELDS            4
#endif
// Default number of mapper slots
#ifndef DEFAULT_MAPPER_SLOTS
#define DEFAULT_MAPPER_SLOTS            8
#endif
// Default number of contexts made for each runtime instance
// Ideally this is a power of 2 (better for performance)
#ifndef DEFAULT_CONTEXTS
#define DEFAULT_CONTEXTS                8
#endif
// Maximum number of sub-tasks per task at a time
#ifndef DEFAULT_MAX_TASK_WINDOW
#define DEFAULT_MAX_TASK_WINDOW         1024
#endif
// Default amount of hysteresis on the task window in the
// form of a percentage (must be between 0 and 100)
#ifndef DEFAULT_TASK_WINDOW_HYSTERESIS
#define DEFAULT_TASK_WINDOW_HYSTERESIS  75
#endif
// How many tasks to group together for runtime operations
#ifndef DEFAULT_MIN_TASKS_TO_SCHEDULE
#define DEFAULT_MIN_TASKS_TO_SCHEDULE   32
#endif
// Scheduling granularity for how many operations to
// handle at a time at each stage of the pipeline
#ifndef DEFAULT_SUPERSCALAR_WIDTH
#define DEFAULT_SUPERSCALAR_WIDTH       4
#endif
// The maximum size of active messages sent by the runtime in bytes
// Note this value was picked based on making a tradeoff between
// latency and bandwidth numbers on both Cray and Infiniband
// interconnect networks.
#ifndef DEFAULT_MAX_MESSAGE_SIZE
#define DEFAULT_MAX_MESSAGE_SIZE        16384
#endif
// Timeout before checking for whether a logical user
// should be pruned from the logical region tree data strucutre
// Making the value less than or equal to zero will
// result in checks always being performed
#ifndef DEFAULT_LOGICAL_USER_TIMEOUT
#define DEFAULT_LOGICAL_USER_TIMEOUT    32
#endif
// Number of events to place in each GC epoch
// Large counts improve efficiency but add latency to
// garbage collection.  Smaller count reduce efficiency
// but improve latency of collection.
#ifndef DEFAULT_GC_EPOCH_SIZE
#define DEFAULT_GC_EPOCH_SIZE           64
#endif

// Used for debugging memory leaks
// How often tracing information is dumped
// based on the number of scheduler invocations
#ifndef TRACE_ALLOCATION_FREQUENCY
#define TRACE_ALLOCATION_FREQUENCY      1024
#endif

// The maximum alignment guaranteed on the 
// target machine bytes.  For most 64-bit 
// systems this should be 16 bytes.
#ifndef LEGION_MAX_ALIGNMENT
#define LEGION_MAX_ALIGNMENT            16
#endif

// Give an ideal upper bound on the maximum
// number of operations Legion should keep
// available for recycling. Where possible
// the runtime will delete objects to keep
// overall memory usage down.
#ifndef LEGION_MAX_RECYCLABLE_OBJECTS
#define LEGION_MAX_RECYCLABLE_OBJECTS      1024
#endif

// An initial seed for random numbers
// generated by the high-level runtime.
#ifndef LEGION_INIT_SEED
#define LEGION_INIT_SEED                  0x221B
#endif

// The radix for the runtime to use when 
// performing collective operations internally
#ifndef LEGION_COLLECTIVE_RADIX
#define LEGION_COLLECTIVE_RADIX           8
#endif

// The radix for the broadcast tree
// when attempting to shutdown the runtime
#ifndef LEGION_SHUTDOWN_RADIX
#define LEGION_SHUTDOWN_RADIX             8
#endif

// Some helper macros

// This statically computes an integer log base 2 for a number
// which is guaranteed to be a power of 2. Adapted from
// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
#define STATIC_LOG2(x)  (LOG2_LOOKUP((uint32_t)(x * 0x077CB531U) >> 27))
#define LOG2_LOOKUP(x) ((x==0) ? 0 : (x==1) ? 1 : (x==2) ? 28 : (x==3) ? 2 : \
                   (x==4) ? 29 : (x==5) ? 14 : (x==6) ? 24 : (x==7) ? 3 : \
                   (x==8) ? 30 : (x==9) ? 22 : (x==10) ? 20 : (x==11) ? 15 : \
                   (x==12) ? 25 : (x==13) ? 17 : (x==14) ? 4 : (x==15) ? 8 : \
                   (x==16) ? 31 : (x==17) ? 27 : (x==18) ? 13 : (x==19) ? 23 : \
                   (x==20) ? 21 : (x==21) ? 19 : (x==22) ? 16 : (x==23) ? 7 : \
                   (x==24) ? 26 : (x==25) ? 12 : (x==26) ? 18 : (x==27) ? 6 : \
                   (x==28) ? 11 : (x==29) ? 5 : (x==30) ? 10 : 9)

#ifndef LEGION_FIELD_LOG2
#define LEGION_FIELD_LOG2         STATIC_LOG2(MAX_FIELDS) // log2(MAX_FIELDS)
#endif

#define LEGION_STRINGIFY(x) #x
#define LEGION_MACRO_TO_STRING(x) LEGION_STRINGIFY(x)

#define LEGION_DISTRIBUTED_ID_MASK    0x00FFFFFFFFFFFFFFULL
#define LEGION_DISTRIBUTED_ID_FILTER(x) ((x) & 0x00FFFFFFFFFFFFFFULL)
#define LEGION_DISTRIBUTED_HELP_DECODE(x)   ((x) >> 56)
#define LEGION_DISTRIBUTED_HELP_ENCODE(x,y) ((x) | ((y) << 56))

// The following enums are all re-exported by
// namespace Legion. These versions are here to facilitate the
// C API. If you are writing C++ code, use the namespaced versions.

typedef enum legion_error_t {
  NO_ERROR = 0,
  ERROR_RESERVED_REDOP_ID = 1,
  ERROR_DUPLICATE_REDOP_ID = 2,
  ERROR_RESERVED_TYPE_HANDLE = 3,
  ERROR_DUPLICATE_TYPE_HANDLE = 4,
  ERROR_DUPLICATE_FIELD_ID = 5,
  ERROR_PARENT_TYPE_HANDLE_NONEXISTENT = 6,
  ERROR_MISSING_PARENT_FIELD_ID = 7,
  ERROR_RESERVED_PROJECTION_ID = 8,
  ERROR_DUPLICATE_PROJECTION_ID = 9,
  ERROR_UNREGISTERED_VARIANT = 10,
  ERROR_USE_REDUCTION_REGION_REQ = 11,
  ERROR_INVALID_ACCESSOR_REQUESTED = 12,
  ERROR_PHYSICAL_REGION_UNMAPPED = 13,
  ERROR_RESERVED_TASK_ID = 14,
  ERROR_INVALID_ARG_MAP_DESTRUCTION = 15,
  ERROR_RESERVED_MAPPING_ID = 16,
  ERROR_BAD_INDEX_PRIVILEGES = 17,
  ERROR_BAD_FIELD_PRIVILEGES = 18,
  ERROR_BAD_REGION_PRIVILEGES = 19,
  ERROR_BAD_PARTITION_PRIVILEGES = 20,
  ERROR_BAD_PARENT_INDEX = 21,
  ERROR_BAD_INDEX_PATH = 22,
  ERROR_BAD_PARENT_REGION = 23,
  ERROR_BAD_REGION_PATH = 24,
  ERROR_BAD_PARTITION_PATH = 25,
  ERROR_BAD_FIELD = 26,
  ERROR_BAD_REGION_TYPE = 27,
  ERROR_INVALID_TYPE_HANDLE = 28,
  ERROR_LEAF_TASK_VIOLATION = 29,
  ERROR_INVALID_REDOP_ID = 30,
  ERROR_REDUCTION_INITIAL_VALUE_MISMATCH = 31,
  ERROR_INVALID_UNMAP_OP = 32,
  ERROR_INVALID_DUPLICATE_MAPPING = 33,
  ERROR_INVALID_REGION_ARGUMENT_INDEX = 34,
  ERROR_INVALID_MAPPING_ACCESS = 35,
  ERROR_STALE_INLINE_MAPPING_ACCESS = 36,
  ERROR_INVALID_INDEX_SPACE_PARENT = 37,
  ERROR_INVALID_INDEX_PART_PARENT = 38,
  ERROR_INVALID_INDEX_SPACE_COLOR = 39,
  ERROR_INVALID_INDEX_PART_COLOR = 40,
  ERROR_INVALID_INDEX_SPACE_HANDLE = 41,
  ERROR_INVALID_INDEX_PART_HANDLE = 42,
  ERROR_FIELD_SPACE_FIELD_MISMATCH = 43,
  ERROR_INVALID_INSTANCE_FIELD = 44,
  ERROR_DUPLICATE_INSTANCE_FIELD = 45,
  ERROR_TYPE_INST_MISMATCH = 46,
  ERROR_TYPE_INST_MISSIZE = 47,
  ERROR_INVALID_INDEX_SPACE_ENTRY = 48,
  ERROR_INVALID_INDEX_PART_ENTRY = 49,
  ERROR_INVALID_FIELD_SPACE_ENTRY = 50,
  ERROR_INVALID_REGION_ENTRY = 51,
  ERROR_INVALID_PARTITION_ENTRY = 52,
  ERROR_ALIASED_INTRA_TASK_REGIONS = 53,
  ERROR_MAX_FIELD_OVERFLOW = 54,
  ERROR_MISSING_TASK_COLLECTION = 55,
  ERROR_INVALID_IDENTITY_PROJECTION_USE = 56,
  ERROR_INVALID_PROJECTION_ID = 57,
  ERROR_NON_DISJOINT_PARTITION = 58,
  ERROR_BAD_PROJECTION_USE = 59,
  ERROR_INDEPENDENT_SLICES_VIOLATION = 60,
  ERROR_INVALID_REGION_HANDLE = 61,
  ERROR_INVALID_PARTITION_HANDLE = 62,
  ERROR_VIRTUAL_MAP_IN_LEAF_TASK = 63,
  ERROR_LEAF_MISMATCH = 64,
  ERROR_INVALID_PROCESSOR_SELECTION = 65,
  ERROR_INVALID_VARIANT_SELECTION = 66,
  ERROR_INVALID_MAPPER_OUTPUT = 67,
  ERROR_UNINITIALIZED_REDUCTION = 68,
  ERROR_INVALID_INDEX_DOMAIN = 69,
  ERROR_INVALID_INDEX_PART_DOMAIN = 70,
  ERROR_DISJOINTNESS_TEST_FAILURE = 71,
  ERROR_NON_DISJOINT_TASK_REGIONS = 72,
  ERROR_INVALID_FIELD_ACCESSOR_PRIVILEGES = 73,
  ERROR_INVALID_PREMAPPED_REGION_LOCATION = 74,
  ERROR_IDEMPOTENT_MISMATCH = 75,
  ERROR_INVALID_MAPPER_ID = 76,
  ERROR_INVALID_TREE_ENTRY = 77,
  ERROR_SEPARATE_UTILITY_PROCS = 78,
  ERROR_MAXIMUM_NODES_EXCEEDED = 79,
  ERROR_MAXIMUM_PROCS_EXCEEDED = 80,
  ERROR_INVALID_TASK_ID = 81,
  ERROR_INVALID_MAPPER_DOMAIN_SLICE = 82,
  ERROR_UNFOLDABLE_REDUCTION_OP = 83,
  ERROR_INVALID_INLINE_ID = 84,
  ERROR_ILLEGAL_MUST_PARALLEL_INLINE = 85,
  ERROR_RETURN_SIZE_MISMATCH = 86,
  ERROR_ACCESSING_EMPTY_FUTURE = 87,
  ERROR_ILLEGAL_PREDICATE_FUTURE = 88,
  ERROR_COPY_REQUIREMENTS_MISMATCH = 89,
  ERROR_INVALID_COPY_FIELDS_SIZE = 90,
  ERROR_COPY_SPACE_MISMATCH = 91,
  ERROR_INVALID_COPY_PRIVILEGE = 92,
  ERROR_INVALID_PARTITION_COLOR = 93,
  ERROR_EXCEEDED_MAX_CONTEXTS = 94,
  ERROR_ACQUIRE_MISMATCH = 95,
  ERROR_RELEASE_MISMATCH = 96,
  ERROR_INNER_LEAF_MISMATCH = 97,
  ERROR_INVALID_FIELD_PRIVILEGES = 98,
  ERROR_ILLEGAL_NESTED_TRACE = 99,
  ERROR_UNMATCHED_END_TRACE = 100,
  ERROR_CONFLICTING_PARENT_MAPPING_DEADLOCK = 101,
  ERROR_CONFLICTING_SIBLING_MAPPING_DEADLOCK = 102,
  ERROR_INVALID_PARENT_REQUEST = 103,
  ERROR_INVALID_FIELD_ID = 104,
  ERROR_NESTED_MUST_EPOCH = 105,
  ERROR_UNMATCHED_MUST_EPOCH = 106,
  ERROR_MUST_EPOCH_FAILURE = 107,
  ERROR_DOMAIN_DIM_MISMATCH = 108,
  ERROR_INVALID_PROCESSOR_NAME = 109,
  ERROR_INVALID_INDEX_SUBSPACE_REQUEST = 110,
  ERROR_INVALID_INDEX_SUBPARTITION_REQUEST = 111,
  ERROR_INVALID_FIELD_SPACE_REQUEST = 112,
  ERROR_INVALID_LOGICAL_SUBREGION_REQUEST = 113,
  ERROR_INVALID_LOGICAL_SUBPARTITION_REQUEST = 114,
  ERROR_ALIASED_REGION_REQUIREMENTS = 115,
  ERROR_MISSING_DEFAULT_PREDICATE_RESULT = 116,
  ERROR_PREDICATE_RESULT_SIZE_MISMATCH = 117,
  ERROR_MPI_INTEROPERABILITY_NOT_CONFIGURED = 118,
  ERROR_TRACING_ALLOCATION_WITH_SEPARATE = 119,
  ERROR_EMPTY_INDEX_PARTITION = 120,
  ERROR_INCONSISTENT_SEMANTIC_TAG = 121,
  ERROR_INVALID_SEMANTIC_TAG = 122,
  ERROR_DUMMY_CONTEXT_OPERATION = 123,
  ERROR_INVALID_CONTEXT_CONFIGURATION = 124,
  ERROR_INDEX_TREE_MISMATCH = 125,
  ERROR_INDEX_PARTITION_ANCESTOR = 126,
  ERROR_INVALID_PENDING_CHILD = 127,
  ERROR_ILLEGAL_FILE_ATTACH = 128,
  ERROR_ILLEGAL_ALLOCATOR_REQUEST = 129,
  ERROR_ILLEGAL_DETACH_OPERATION = 130,
  ERROR_NO_PROCESSORS = 131,
  ERROR_ILLEGAL_REDUCTION_VIRTUAL_MAPPING = 132,
  ERROR_INVALID_MAPPED_REGION_LOCATION = 133,
  ERROR_RESERVED_SERDEZ_ID = 134,
  ERROR_DUPLICATE_SERDEZ_ID = 135,
  ERROR_INVALID_SERDEZ_ID = 136,
  ERROR_TRACE_VIOLATION = 137,
  ERROR_INVALID_TARGET_PROC = 138,
  ERROR_INCOMPLETE_TRACE = 139,
  ERROR_STATIC_CALL_POST_RUNTIME_START = 140,
  ERROR_ILLEGAL_GLOBAL_VARIANT_REGISTRATION = 141,
  ERROR_ILLEGAL_USE_OF_NON_GLOBAL_VARIANT = 142,
  ERROR_RESERVED_CONSTRAINT_ID = 143,
  ERROR_INVALID_CONSTRAINT_ID = 144,
  ERROR_DUPLICATE_CONSTRAINT_ID = 145,
  ERROR_ILLEGAL_WAIT_FOR_SHUTDOWN = 146,
  ERROR_DEPRECATED_METHOD_USE = 147,
  ERROR_MAX_APPLICATION_TASK_ID_EXCEEDED = 148,
  ERROR_MAX_APPLICATION_MAPPER_ID_EXCEEDED = 149,
  ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME = 150,
  ERROR_INVALID_MAPPER_SYNCHRONIZATION = 151,
  ERROR_ILLEGAL_PARTIAL_ACQUISITION = 152,
  ERROR_ILLEGAL_INTERFERING_RESTRICTIONS = 153,
  ERROR_ILLEGAL_PARTIAL_RESTRICTION = 154,
  ERROR_ILLEGAL_INTERFERING_ACQUISITIONS = 155,
  ERROR_UNRESTRICTED_ACQUIRE = 156,
  ERROR_UNACQUIRED_RELEASE = 157,
  ERROR_UNATTACHED_DETACH = 158,
  ERROR_INVALID_PROJECTION_RESULT = 159,
  ERROR_ILLEGAL_IMPLICIT_MAPPING = 160,
  ERROR_INNER_TASK_VIOLATION = 161,
  ERROR_REQUEST_FOR_EMPTY_FUTURE = 162,
  ERROR_ILLEGAL_REMAP_IN_STATIC_TRACE = 163,
  ERROR_MISSING_LOCAL_VARIABLE = 164,
}  legion_error_t;

// enum and namepsaces don't really get along well
typedef enum legion_privilege_mode_t {
  NO_ACCESS       = 0x00000000, 
  READ_ONLY       = 0x00000001,
  READ_WRITE      = 0x00000007, // All three privileges
  WRITE_ONLY      = 0x00000002, // same as WRITE_DISCARD
  WRITE_DISCARD   = 0x00000002, // same as WRITE_ONLY
  REDUCE          = 0x00000004,
} legion_privilege_mode_t;

typedef enum legion_allocate_mode_t {
  NO_MEMORY       = 0x00000000,
  ALLOCABLE       = 0x00000001,
  FREEABLE        = 0x00000002,
  MUTABLE         = 0x00000003,
  REGION_CREATION = 0x00000004,
  REGION_DELETION = 0x00000008,
  ALL_MEMORY      = 0x0000000F,
} legion_allocate_mode_t;

typedef enum legion_coherence_property_t {
  EXCLUSIVE    = 0,
  ATOMIC       = 1,
  SIMULTANEOUS = 2,
  RELAXED      = 3,
} legion_coherence_property_t;

// Optional region requirement flags
typedef enum legion_region_flags_t {
  NO_FLAG         = 0x00000000,
  VERIFIED_FLAG   = 0x00000001,
  NO_ACCESS_FLAG  = 0x00000002, // Deprecated, user SpecializedConstraint
  RESTRICTED_FLAG = 0x00000004,
  MUST_PREMAP_FLAG= 0x00000008,
} legion_region_flags_t;

typedef enum legion_index_space_kind_t {
  UNSTRUCTURED_KIND,
  SPARSE_ARRAY_KIND,
  DENSE_ARRAY_KIND,
} legion_index_space_kind_t;

typedef enum legion_projection_type_t {
  SINGULAR, // a single logical region
  PART_PROJECTION, // projection from a partition
  REG_PROJECTION, // projection from a region
} legion_projection_type_t;
// For backwards compatibility
typedef legion_projection_type_t legion_handle_type_t;

typedef enum legion_partition_kind_t {
  DISJOINT_KIND,
  ALIASED_KIND,
  COMPUTE_KIND,
} legion_partition_kind_t;

typedef enum legion_external_resource_t {
  EXTERNAL_POSIX_FILE,
  EXTERNAL_HDF5_FILE,
  EXTERNAL_C_ARRAY,
  EXTERNAL_FORTRAN_ARRAY,
} legion_external_resource_t;

typedef enum legion_timing_measurement_t {
  MEASURE_SECONDS,
  MEASURE_MICRO_SECONDS,
  MEASURE_NANO_SECONDS,
} legion_timing_measurement_t;

typedef enum legion_dependence_type_t {
  NO_DEPENDENCE = 0,
  TRUE_DEPENDENCE = 1,
  ANTI_DEPENDENCE = 2, // WAR or WAW with Write-Only privilege
  ATOMIC_DEPENDENCE = 3,
  SIMULTANEOUS_DEPENDENCE = 4,
} legion_dependence_type_t;

enum {
  NAME_SEMANTIC_TAG = 0,
  FIRST_AVAILABLE_SEMANTIC_TAG = 1,
};

typedef enum legion_execution_constraint_t {
  ISA_CONSTRAINT = 0, // instruction set architecture
  PROCESSOR_CONSTRAINT = 1, // processor kind constraint
  RESOURCE_CONSTRAINT = 2, // physical resources
  LAUNCH_CONSTRAINT = 3, // launch configuration
  COLOCATION_CONSTRAINT = 4, // region requirements in same instance
} legion_execution_constraint_t;

typedef enum legion_layout_constraint_t {
  SPECIALIZED_CONSTRAINT = 0, // normal or speicalized (e.g. reduction-fold)
  MEMORY_CONSTRAINT = 1, // constraint on the kind of memory
  FIELD_CONSTRAINT = 2, // ordering of fields
  ORDERING_CONSTRAINT = 3, // ordering of dimensions
  SPLITTING_CONSTRAINT = 4, // splitting of dimensions 
  DIMENSION_CONSTRAINT = 5, // dimension size constraint
  ALIGNMENT_CONSTRAINT = 6, // alignment of a field
  OFFSET_CONSTRAINT = 7, // offset of a field
  POINTER_CONSTRAINT = 8, // pointer of a field
} legion_layout_constraint_t;

typedef enum legion_equality_kind_t {
  LT_EK = 0, // <
  LE_EK = 1, // <=
  GT_EK = 2, // >
  GE_EK = 3, // >=
  EQ_EK = 4, // ==
  NE_EK = 5, // !=
} legion_equality_kind_t;

typedef enum legion_dimension_kind_t {
  DIM_X = 0, // first logical index space dimension
  DIM_Y = 1, // second logical index space dimension
  DIM_Z = 2, // ...
  DIM_F = 3, // field dimension
  INNER_DIM_X = 4, // inner dimension for tiling X
  OUTER_DIM_X = 5, // outer dimension for tiling X
  INNER_DIM_Y = 6, // ...
  OUTER_DIM_Y = 7,
  INNER_DIM_Z = 8,
  OUTER_DIM_Z = 9,
} legion_dimension_kind_t;

// Make all flags 1-hot encoding so we can logically-or them together
typedef enum legion_isa_kind_t {
  // Top-level ISA Kinds
  X86_ISA   = 0x00000001,
  ARM_ISA   = 0x00000002,
  POW_ISA   = 0x00000004, // Power PC
  PTX_ISA   = 0x00000008, // auto-launch by runtime
  CUDA_ISA  = 0x00000010, // run on CPU thread bound to CUDA context
  LUA_ISA   = 0x00000020, // run on Lua processor
  TERRA_ISA = 0x00000040, // JIT to target processor kind
  LLVM_ISA  = 0x00000080, // JIT to target processor kind
  GL_ISA    = 0x00000100, // run on CPU thread with OpenGL context
  // x86 Vector Instructions
  SSE_ISA   = 0x00000200,
  SSE2_ISA  = 0x00000400,
  SSE3_ISA  = 0x00000800,
  SSE4_ISA  = 0x00001000,
  AVX_ISA   = 0x00002000,
  AVX2_ISA  = 0x00004000,
  FMA_ISA   = 0x00008000,
  MIC_ISA   = 0x00010000,
  // GPU variants
  SM_10_ISA = 0x00020000,
  SM_20_ISA = 0x00040000,
  SM_30_ISA = 0x00080000,
  SM_35_ISA = 0x00100000,
  // ARM Vector Instructions
  NEON_ISA  = 0x00200000,
} legion_isa_kind_t;

typedef enum legion_resource_constraint_t {
  L1_CACHE_SIZE = 0,
  L2_CACHE_SIZE = 1,
  L3_CACHE_SIZE = 2,
  L1_CACHE_ASSOCIATIVITY = 3,
  L2_CACHE_ASSOCIATIVITY = 4,
  L3_CACHE_ASSOCIATIVITY = 5,
  REGISTER_FILE_SIZE = 6,
  SHARED_MEMORY_SIZE = 7,
  TEXTURE_CACHE_SIZE = 8,
  CONSTANT_CACHE_SIZE = 9,
  NAMED_BARRIERS = 10,
  SM_COUNT = 11, // total SMs on the device
  MAX_OCCUPANCY = 12, // max warps per SM
} legion_resource_constraint_t;

typedef enum legion_launch_constraint_t {
  CTA_SHAPE = 0,
  GRID_SHAPE = 1,
  DYNAMIC_SHARED_MEMORY = 2,
  REGISTERS_PER_THREAD = 3,
  CTAS_PER_SM = 4,
  NAMED_BARRIERS_PER_CTA = 5,
} legion_launch_constraint_t;

typedef enum legion_specialized_constraint_t {
  NO_SPECIALIZE = 0,
  NORMAL_SPECIALIZE = 1,
  REDUCTION_FOLD_SPECIALIZE = 2,
  REDUCTION_LIST_SPECIALIZE = 3,
  VIRTUAL_SPECIALIZE = 4,
  // All file types must go below here, everything else above
  GENERIC_FILE_SPECIALIZE = 5,
  HDF5_FILE_SPECIALIZE = 6,
} legion_specialized_constraint_t;

//==========================================================================
//                                Types
//==========================================================================

typedef legion_lowlevel_file_mode_t legion_file_mode_t;
typedef legion_lowlevel_processor_kind_t legion_processor_kind_t;
typedef legion_lowlevel_memory_kind_t legion_memory_kind_t;
typedef legion_lowlevel_domain_max_rect_dim_t legion_domain_max_rect_dim_t;
typedef legion_lowlevel_reduction_op_id_t legion_reduction_op_id_t;
typedef legion_lowlevel_custom_serdez_id_t legion_custom_serdez_id_t;
typedef legion_lowlevel_address_space_t legion_address_space_t;
typedef int legion_task_priority_t;
typedef int legion_garbage_collection_priority_t;
typedef unsigned int legion_color_t;
typedef unsigned int legion_field_id_t;
typedef unsigned int legion_trace_id_t;
typedef unsigned int legion_mapper_id_t;
typedef unsigned int legion_context_id_t;
typedef unsigned int legion_instance_id_t;
typedef unsigned int legion_index_space_id_t;
typedef unsigned int legion_index_partition_id_t;
typedef unsigned int legion_index_tree_id_t;
typedef unsigned int legion_field_space_id_t;
typedef unsigned int legion_generation_id_t;
typedef unsigned int legion_type_handle;
typedef unsigned int legion_projection_id_t;
typedef unsigned int legion_region_tree_id_t;
typedef unsigned int legion_address_space_id_t;
typedef unsigned int legion_tunable_id_t;
typedef unsigned int legion_local_variable_id_t;
typedef unsigned int legion_generator_id_t;
typedef unsigned long long legion_distributed_id_t;
typedef unsigned long legion_mapping_tag_id_t;
typedef unsigned long legion_variant_id_t;
typedef unsigned long legion_semantic_tag_t;
typedef unsigned long long legion_unique_id_t;
typedef unsigned long long legion_version_id_t;
typedef unsigned long long legion_projection_epoch_id_t;
typedef legion_lowlevel_task_func_id_t legion_task_id_t;
typedef unsigned long legion_layout_constraint_id_t;

#endif // __LEGION_CONFIG_H__

