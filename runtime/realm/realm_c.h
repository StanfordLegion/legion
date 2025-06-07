/* Copyright 2025 Stanford University, NVIDIA Corporation
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

// C-only header for Realm - mostly includes typedefs right now,
//  but may be expanded to provide C bindings for the Realm API

#ifndef REALM_C_H
#define REALM_C_H

#include "realm/realm_config.h"

#ifndef LEGION_USE_PYTHON_CFFI
// for size_t
#include <stddef.h>
// TODO: Fix me, use dllimport / visibility from realm_exports.h
#define REALM_EXPORT REALM_PUBLIC_API
#else
#define REALM_EXPORT
#endif // LEGION_USE_PYTHON_CFFI

// for uint64_t
#include <stdint.h>

#if defined(_WIN32)
// Force a specific calling convention for function pointers
#define REALM_FNPTR __stdcall
#else
#define REALM_FNPTR
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct realm_runtime_st;
typedef struct realm_runtime_st *realm_runtime_t;

struct realm_profiling_request_set_st;
typedef struct realm_profiling_request_set_st *realm_profiling_request_set_t;

struct realm_processor_query_st;
typedef struct realm_processor_query_st *realm_processor_query_t;

struct realm_memory_query_st;
typedef struct realm_memory_query_st *realm_memory_query_t;

typedef unsigned long long realm_id_t;
typedef realm_id_t realm_event_t;
typedef realm_id_t realm_user_event_t;
typedef realm_id_t realm_processor_t;
typedef realm_id_t realm_memory_t;
typedef realm_id_t realm_instance_t;
#define IDFMT "%llx" // TODO: name it to REALM_IDFMT

typedef unsigned int realm_address_space_t;
typedef unsigned realm_task_func_id_t;
typedef int realm_reduction_op_id_t;
typedef int realm_custom_serdez_id_t;
typedef unsigned realm_event_gen_t;
typedef int realm_field_id_t;
typedef unsigned long long realm_barrier_timestamp_t;

#define REALM_NO_PROC ((realm_processor_t)0ULL)
#define REALM_NO_MEM ((realm_memory_t)0ULL)
#define REALM_NO_EVENT ((realm_event_t)0ULL)
#define REALM_NO_USER_EVENT ((realm_user_event_t)0ULL)

#define REALM_TASK_ID_PROCESSOR_NOP ((realm_task_func_id_t)0U)
#define REALM_TASK_ID_PROCESSOR_INIT ((realm_task_func_id_t)1U)
#define REALM_TASK_ID_PROCESSOR_SHUTDOWN ((realm_task_func_id_t)2U)
#define REALM_TASK_ID_FIRST_AVAILABLE ((realm_task_func_id_t)4U)

typedef enum realm_register_task_flags
{
  REALM_REGISTER_TASK_DEFAULT = 0x0ULL,
  REALM_REGISTER_TASK_GLOBAL = 0x1ULL,
  REALM_REGISTER_TASK_NUM,
  REALM_REGISTER_TASK_MAX = 0x7fffffffULL,
} realm_register_task_flags_t;

typedef enum realm_processor_attr_enum
{
  REALM_PROCESSOR_ATTR_KIND = 0x0ULL,
  REALM_PROCESSOR_ATTR_ADDRESS_SPACE,
  REALM_PROCESSOR_ATTR_NUM,
  REALM_PROCESSOR_ATTR_MAX = 0xFFFFFFFFFFFFFFFFULL,
} realm_processor_attr_t;

typedef enum realm_memory_attr_enum
{
  REALM_MEMORY_ATTR_KIND = 0x0ULL,
  REALM_MEMORY_ATTR_ADDRESS_SPACE,
  REALM_MEMORY_ATTR_CAPACITY,
  REALM_MEMORY_ATTR_NUM,
  REALM_MEMORY_ATTR_MAX = 0xFFFFFFFFFFFFFFFFULL,
} realm_memory_attr_t;

// Different Processor types
#define REALM_PROCESSOR_KINDS(__op__) \
  __op__(NO_KIND, "") \
  __op__(TOC_PROC, "Throughput core") \
  __op__(LOC_PROC, "Latency core") \
  __op__(UTIL_PROC, "Utility core") \
  __op__(IO_PROC, "I/O core") \
  __op__(PROC_GROUP, "Processor group") \
  __op__(PROC_SET, "Set of Processors for OpenMP/Kokkos etc.") \
  __op__(OMP_PROC, "OpenMP (or similar) thread pool") \
  __op__(PY_PROC, "Python interpreter")

typedef enum realm_processor_kind_t {
#define C_ENUMS(name, desc) name,
  REALM_PROCESSOR_KINDS(C_ENUMS)
#undef C_ENUMS
} realm_processor_kind_t;

// Different Memory types
#define REALM_MEMORY_KINDS(__op__) \
  __op__(NO_MEMKIND, "") \
  __op__(GLOBAL_MEM, "Guaranteed visible to all processors on all nodes (e.g. GASNet memory, universally slow)") \
  __op__(SYSTEM_MEM, "Visible to all processors on a node") \
  __op__(REGDMA_MEM, "Registered memory visible to all processors on a node, can be a target of RDMA") \
  __op__(SOCKET_MEM, "Memory visible to all processors within a node, better performance to processors on same socket") \
  __op__(Z_COPY_MEM, "Zero-Copy memory visible to all CPUs within a node and one or more GPUs") \
  __op__(GPU_FB_MEM, "Framebuffer memory for one GPU and all its SMs") \
  __op__(DISK_MEM, "Disk memory visible to all processors on a node") \
  __op__(HDF_MEM, "HDF memory visible to all processors on a node") \
  __op__(FILE_MEM, "file memory visible to all processors on a node") \
  __op__(LEVEL3_CACHE, "CPU L3 Visible to all processors on the node, better performance to processors on same socket") \
  __op__(LEVEL2_CACHE, "CPU L2 Visible to all processors on the node, better performance to one processor") \
  __op__(LEVEL1_CACHE, "CPU L1 Visible to all processors on the node, better performance to one processor") \
  __op__(GPU_MANAGED_MEM, "Managed memory that can be cached by either host or GPU") \
  __op__(GPU_DYNAMIC_MEM, "Dynamically-allocated framebuffer memory for one GPU and all its SMs")

typedef enum realm_memory_kind_t {
#define C_ENUMS(name, desc) name,
  REALM_MEMORY_KINDS(C_ENUMS)
#undef C_ENUMS
} realm_memory_kind_t;

// file modes - to be removed soon
typedef enum realm_file_mode_t
{
  REALM_FILE_READ_ONLY,
  REALM_FILE_READ_WRITE,
  REALM_FILE_CREATE,
  // These are deprecated but maintained for backwards compatibility
  LEGION_FILE_READ_ONLY = REALM_FILE_READ_ONLY,
  LEGION_FILE_READ_WRITE = REALM_FILE_READ_WRITE,
  LEGION_FILE_CREATE = REALM_FILE_CREATE,
} realm_file_mode_t;

// Prototype for a Realm task
typedef void(REALM_FNPTR *realm_task_pointer_t)(const void * /*data*/, size_t /*datalen*/,
                                                const void * /*userdata*/,
                                                size_t /*userlen*/,
                                                realm_processor_t /*proc_id*/);

// error code
typedef enum realm_status_enum
{
  // To comply with other C libraries, we use 0 for success, negative numbers for errors
  REALM_SUCCESS = 0,
  REALM_ERROR = -1,
  REALM_ARGUMENT_ERROR_WITH_EXTRA_FLAGS =
      -1000, // this is a soft error, the caller will expect to receive correct results
  REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER = -1001,
  REALM_ARGUMENT_ERROR_UNKNOWN_INTEGER_UNIT = -1002,
  REALM_ARGUMENT_ERROR_MISSING_INPUT = -1003,
  REALM_ARGUMENT_ERROR_OUTPUT_STRING_TOO_SHORT = -1004,
  REALM_ARGUMENT_ERROR_METHOD_RETURN_FALSE = -1005,
  REALM_TOPOLOGY_ERROR_NO_AFFINITY = -2001,
  REALM_TOPOLOGY_ERROR_LINUX_NO_CPU_DIR = -2002,
  REALM_TOPOLOGY_ERROR_LINUX_NO_NUMA_DIR = -2003,
  REALM_TOPOLOGY_ERROR_HWLOC_INIT_FAILED = -2004,
  REALM_TOPOLOGY_ERROR_HWLOC_LOAD_TOPO_FAILED = -2005,
  REALM_TOPOLOGY_ERROR_HWLOC_TYPE_DEPTH_UNKNOWN = -2006,
  REALM_TOPOLOGY_ERROR_ENV_LOAD_FAILED = -2007,
  REALM_TOPOLOGY_ERROR_WIN32_NO_PROC_INFO = -2008,
  REALM_RUNTIME_ERROR_INVALID_RUNTIME = -3001,
  REALM_RUNTIME_ERROR_NOT_INITIALIZED = -3002,
  REALM_MACHINE_ERROR_INVALID_MACHINE = -4001,
  REALM_MEMORY_ERROR_INVALID_MEMORY = -5001,
  REALM_PROCESSOR_ERROR_INVALID_PROCESSOR = -6001,
  REALM_PROCESSOR_ERROR_INVALID_PROCESSOR_KIND = -6002,
  REALM_PROCESSOR_ERROR_INVALID_ATTRIBUTE = -6003,
  REALM_PROCESSOR_ERROR_INVALID_TASK_FUNCTION = -6004,
  REALM_MEMORY_ERROR_INVALID_MEMORY_KIND = -7001,
  REALM_MEMORY_ERROR_INVALID_ATTRIBUTE = -7002,
  REALM_EVENT_ERROR_INVALID_EVENT = -8001,
  REALM_PROCESSOR_QUERY_ERROR_INVALID_QUERY = -9001,
  REALM_PROCESSOR_QUERY_ERROR_INVALID_CALLBACK = -9002,
  REALM_MEMORY_QUERY_ERROR_INVALID_QUERY = -10001,
  REALM_MEMORY_QUERY_ERROR_INVALID_CALLBACK = -10002,
  REALM_ADDRESS_SPACE_INVALID = -11001
} realm_status_t;

typedef realm_status_t RealmStatus;

// Callback function for processor query iteration.
typedef realm_status_t (*REALM_FNPTR realm_processor_query_cb_t)(realm_processor_t /*p*/,
                                                                 void * /*user_data*/);

// Callback function for memory query iteration.
typedef realm_status_t (*REALM_FNPTR realm_memory_query_cb_t)(realm_memory_t /*m*/,
                                                              void * /*user_data*/);

/*
 * @defgroup Runtime Runtime API
 * @ingroup Realm
 */

/**
 * @brief Creates a new Realm runtime instance.
 *
 * @param[out] runtime A pointer to the runtime instance to be created.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Runtime
 */
realm_status_t REALM_EXPORT realm_runtime_create(realm_runtime_t *runtime);

/**
 * @brief Destroys a Realm runtime instance, please make sure all works are finished and
 * the runtime has been shutdown before destroying the runtime.
 *
 * @param runtime The runtime instance to be destroyed.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Runtime
 */
realm_status_t REALM_EXPORT realm_runtime_destroy(realm_runtime_t runtime);

/**
 * @brief Returns the current Realm runtime instance.
 *
 * @param[out] runtime A pointer to the runtime instance to be returned.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Runtime
 */
realm_status_t REALM_EXPORT realm_runtime_get_runtime(realm_runtime_t *runtime);

/**
 * @brief Creates and initializes the Realm runtime with command-line arguments.
 *
 * @param runtime A pointer to the runtime instance to be initialized.
 * @param[in,out] argc A pointer to the argument count.
 * @param[in,out] argv A pointer to the argument vector.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Runtime
 */
realm_status_t REALM_EXPORT realm_runtime_init(realm_runtime_t runtime, int *argc,
                                               char ***argv);

/**
 * @brief Shuts down the Realm runtime.
 *
 * @param runtime The runtime instance to be shut down.
 * @param wait_on An event to wait on before shutting down.
 * @param result_code The result code to return upon shutdown.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Runtime
 */
realm_status_t REALM_EXPORT realm_runtime_signal_shutdown(realm_runtime_t runtime,
                                                          realm_event_t wait_on,
                                                          int result_code);

/**
 * @brief Waits for the Realm runtime to shut down.
 *
 * @param runtime The runtime instance to wait for.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Runtime
 */
realm_status_t REALM_EXPORT realm_runtime_wait_for_shutdown(realm_runtime_t runtime);

/**
 * @brief Spawns the task registered by the task id by \p task_id on the processor given
 * by \p target_proc.  This call has special meaning for multi-process realms as all
 * processes calling this will perform a barrier and a single arbitrary process will be
 * elected to launch the task on the processor.  It is undefined if these arguments do not
 * match across all processes calls in a multi-process invocation.
 *
 * @param runtime The runtime instance to use.
 * @param target_proc The target processor for the task.
 * @param task_id The ID of the task to be spawned.
 * @param args This is a buffer of bytes of length \p arglen to be copied to the executor
 * of the given \p task_id.
 * @param arglen The length of the arguments.
 * @param wait_on An event to wait on before spawning the task.
 * @param priority The priority of the task.
 * @param[out] event An event to signal upon task completion.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Runtime
 */
realm_status_t REALM_EXPORT realm_runtime_collective_spawn(
    realm_runtime_t runtime, realm_processor_t target_proc, realm_task_func_id_t task_id,
    const void *args, size_t arglen, realm_event_t wait_on, int priority,
    realm_event_t *event);

/*
 * @defgroup Processor Processor API
 * @ingroup Realm
 */

/**
 * @brief Registers a task with all processors whose kind matches that of \p target_kind
 *
 * @param runtime The runtime instance to use.
 * @param target_kind The kind of processor to register the task with.
 * @param global Registrations that are global are ones in which the registration only has
 * to be done by one process in a multi-process realm.  This has the requirement that the
 * function is referenceable through calls such as dladdr and dlsym and that the function
 * has the same offset across all processes in the realm.  This usually requires the
 * application to compile with -rdynamic.
 * @param task_id The ID of the task to be registered.
 * @param func The function pointer for the task.
 * @param user_data User data to pass to the task.
 * @param user_data_len The length of the user data.
 * @param[out] event An event to signal upon task registration.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Processor
 */
realm_status_t REALM_EXPORT realm_processor_register_task_by_kind(
    realm_runtime_t runtime, realm_processor_kind_t target_kind,
    realm_register_task_flags_t flags, realm_task_func_id_t task_id,
    realm_task_pointer_t func, void *user_data, size_t user_data_len,
    realm_event_t *event);

/**
 * @brief Spawns a task on a specific processor.
 *
 * @param runtime The runtime instance to use.
 * @param target_proc The target processor for the task.
 * @param task_id The ID of the task to be spawned.
 * @param args The arguments for the task.
 * @param arglen The length of the arguments.
 * @param prs The profiling request set.
 * @param wait_on An event to wait on before spawning the task.
 * @param priority The priority of the task.
 * @param[out] event An event to signal upon task completion.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Processor
 */
realm_status_t REALM_EXPORT realm_processor_spawn(
    realm_runtime_t runtime, realm_processor_t target_proc, realm_task_func_id_t task_id,
    const void *args, size_t arglen, realm_profiling_request_set_t prs,
    realm_event_t wait_on, int priority, realm_event_t *event);

/**
 * @brief Returns the attributes of a processor.
 *
 * @param runtime The runtime instance to use.
 * @param proc The processor to get the attributes of.
 * @param attrs The attributes to get.
 * @param[out] values The values of the attributes.
 * @param num The number of attributes to get.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Processor
 */
realm_status_t REALM_EXPORT realm_processor_get_attributes(realm_runtime_t runtime,
                                                           realm_processor_t proc,
                                                           realm_processor_attr_t *attrs,
                                                           uint64_t *values, size_t num);

/*
 * @defgroup ProcessorQuery ProcessorQuery API
 * @ingroup Realm
 */

/**
 * @brief Creates a new processor query.
 *
 * @param machine The machine instance.
 * @param[out] query The processor query to be created.
 * @return Realm status indicating success or failure.
 *
 * @ingroup ProcessorQuery
 */
realm_status_t REALM_EXPORT realm_processor_query_create(realm_runtime_t runtime,
                                                         realm_processor_query_t *query);

/**
 * @brief Destroys a processor query.
 *
 * @param query The processor query to be destroyed.
 * @return Realm status indicating success or failure.
 *
 * @ingroup ProcessorQuery
 */
realm_status_t REALM_EXPORT realm_processor_query_destroy(realm_processor_query_t query);

/**
 * @brief Restricts the processor query to a specific kind.
 *
 * @param query The processor query to be restricted.
 * @param kind The kind of processor to restrict to.
 * @return Realm status indicating success or failure.
 *
 * @ingroup ProcessorQuery
 */
realm_status_t REALM_EXPORT realm_processor_query_restrict_to_kind(
    realm_processor_query_t query, realm_processor_kind_t kind);

/**
 * @brief Restricts the processor query to address space.
 *
 * @param query The processor query to be restricted.
 * @param address_space The address space to restrict to.
 * @return Realm status indicating success or failure.
 *
 * @ingroup ProcessorQuery
 */
realm_status_t REALM_EXPORT realm_processor_query_restrict_to_address_space(
    realm_processor_query_t query, realm_address_space_t address_space);

/**
 * @brief Iterates over the processors in the query.
 *
 * @param query The processor query to be iterated over.
 * @param cb The callback function to be called for each processor.
 * @param user_data The user data to be passed to the callback function.
 * @param max_queries The maximum number of processors to iterate over.
 * @return Realm status indicating success or failure.
 *
 * @ingroup ProcessorQuery
 */
realm_status_t REALM_EXPORT realm_processor_query_iter(realm_processor_query_t query,
                                                       realm_processor_query_cb_t cb,
                                                       void *user_data,
                                                       size_t max_queries);

/*
 * @defgroup Memory Memory API
 * @ingroup Realm
 */

/**
 * @brief Returns the attributes of a memory.
 *
 * @param runtime The runtime instance to use.
 * @param mem The memory to get the attributes of.
 * @param attrs The attributes to get.
 * @param[out] values The values of the attributes.
 * @param num The number of attributes to get.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Memory
 */
realm_status_t REALM_EXPORT realm_memory_get_attributes(realm_runtime_t runtime,
                                                        realm_memory_t mem,
                                                        realm_memory_attr_t *attrs,
                                                        uint64_t *values, size_t num);

/*
 * @defgroup MemoryQuery MemoryQuery API
 * @ingroup Realm
 */

/**
 * @brief Creates a new memory query.
 *
 * @param machine The machine instance.
 * @param[out] query The memory query to be created.
 * @return Realm status indicating success or failure.
 *
 * @ingroup MemoryQuery
 */
realm_status_t REALM_EXPORT realm_memory_query_create(realm_runtime_t runtime,
                                                      realm_memory_query_t *query);

/**
 * @brief Destroys a memory query.
 *
 * @param query The memory query to be destroyed.
 * @return Realm status indicating success or failure.
 *
 * @ingroup MemoryQuery
 */
realm_status_t REALM_EXPORT realm_memory_query_destroy(realm_memory_query_t query);

/**
 * @brief Restricts the memory query to a specific kind.
 *
 * @param query The memory query to be restricted.
 * @param kind The kind of memory to restrict to.
 * @return Realm status indicating success or failure.
 *
 * @ingroup MemoryQuery
 */
realm_status_t REALM_EXPORT
realm_memory_query_restrict_to_kind(realm_memory_query_t query, realm_memory_kind_t kind);

/**
 * @brief Restricts the memory query to address space.
 *
 * @param query The memory query to be restricted.
 * @param address_space The address space to restrict to.
 * @return Realm status indicating success or failure.
 *
 * @ingroup MemoryQuery
 */
realm_status_t REALM_EXPORT realm_memory_query_restrict_to_address_space(
    realm_memory_query_t query, realm_address_space_t address_space);

/**
 * @brief Restricts the memory query to a minimum capacity.
 *
 * @param query The memory query to be restricted.
 * @param min_bytes The minimum capacity to restrict to.
 * @return Realm status indicating success or failure.
 *
 * @ingroup MemoryQuery
 */
realm_status_t REALM_EXPORT
realm_memory_query_restrict_by_capacity(realm_memory_query_t query, size_t min_bytes);

/**
 * @brief Iterates over the memories in the query.
 *
 * @param query The memory query to be iterated over.
 * @param cb The callback function to be called for each memory.
 * @param user_data The user data to be passed to the callback function.
 * @param max_queries The maximum number of memories to iterate over.
 * @return Realm status indicating success or failure.
 *
 * @ingroup MemoryQuery
 */
realm_status_t REALM_EXPORT realm_memory_query_iter(realm_memory_query_t query,
                                                    realm_memory_query_cb_t cb,
                                                    void *user_data, size_t max_queries);

/*
 * @defgroup Event Event API
 * @ingroup Realm
 */

/**
 * @brief Waits for a specific event to complete.
 *
 * @param runtime The runtime instance to use.
 * @param[out] event The event to wait for.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Event
 */
realm_status_t REALM_EXPORT realm_event_wait(realm_runtime_t runtime,
                                             realm_event_t event);

/**
 * @brief Merges multiple events into a single event.
 *
 * @param runtime The runtime instance to use.
 * @param wait_for The events to wait for.
 * @param num_events The number of events to wait for.
 * @param[out] event The merged event.
 * @return Realm status indicating success or failure.
 *
 * @ingroup Event
 */
realm_status_t REALM_EXPORT realm_event_merge(realm_runtime_t runtime,
                                              const realm_event_t *wait_for,
                                              size_t num_events, realm_event_t *event);

/*
 * @defgroup UserEvent UserEvent API
 * @ingroup Realm
 */

/**
 * @brief Creates a new user event.
 *
 * @param[out] event The user event to be created.
 * @return Realm status indicating success or failure.
 *
 * @ingroup UserEvent
 */
realm_status_t REALM_EXPORT realm_user_event_create(realm_runtime_t runtime,
                                                    realm_user_event_t *event);

/**
 * @brief Triggers a user event.
 *
 * @param event The user event to be triggered.
 * @return Realm status indicating success or failure.
 *
 * @ingroup UserEvent
 */
realm_status_t REALM_EXPORT realm_user_event_trigger(realm_runtime_t runtime,
                                                     realm_user_event_t event);

#ifdef __cplusplus
}
#endif

#endif // ifndef REALM_C_H
