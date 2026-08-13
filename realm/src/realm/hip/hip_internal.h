/*
 * Copyright 2026 Stanford University, NVIDIA Corporation, Los Alamos National Laboratory
 * SPDX-License-Identifier: Apache-2.0
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

#ifndef REALM_HIP_INTERNAL_H
#define REALM_HIP_INTERNAL_H

#include "realm/hip/hip_module.h"

#include <hip/hip_runtime.h>

#include "realm/operation.h"
#include "realm/threads.h"
#include "realm/circ_queue.h"
#include "realm/indexspace.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/bgwork.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"
#include "realm/hip/hip_memcpy.h"
#include "realm/hip/hip_reduc.h"

#define CHECK_CUDART(cmd)                                                                \
  do {                                                                                   \
    hipError_t ret = (cmd);                                                              \
    if(ret != hipSuccess) {                                                              \
      fprintf(stderr, "HIP: %s = %d (%s)\n", #cmd, ret, hipGetErrorString(ret));         \
      assert(0);                                                                         \
      exit(1);                                                                           \
    }                                                                                    \
  } while(0)

#define REPORT_HIP_ERROR(cmd, ret)                                                       \
  do {                                                                                   \
    const char *name, *str;                                                              \
    name = hipGetErrorName(ret);                                                         \
    str = hipGetErrorString(ret);                                                        \
    fprintf(stderr, "HIP: %s = %d (%s): %s\n", cmd, ret, name, str);                     \
    abort();                                                                             \
  } while(0)

#define CHECK_HIP(cmd)                                                                   \
  do {                                                                                   \
    hipError_t ret = (cmd);                                                              \
    if(ret != hipSuccess)                                                                \
      REPORT_HIP_ERROR(#cmd, ret);                                                       \
  } while(0)

namespace Realm {

  namespace Hip {

    struct GPUInfo {
      int index; // index used by HIP runtime
      hipDevice_t device;

      static const size_t MAX_NAME_LEN = 64;
      char name[MAX_NAME_LEN];

      int major, minor;
      size_t totalGlobalMem;
      std::set<hipDevice_t> peers; // other GPUs we can do p2p copies with
    };

    enum GPUMemcpyKind
    {
      GPU_MEMCPY_HOST_TO_DEVICE,
      GPU_MEMCPY_DEVICE_TO_HOST,
      GPU_MEMCPY_DEVICE_TO_DEVICE,
      GPU_MEMCPY_PEER_TO_PEER,
    };

    // Forard declaration
    class GPUProcessor;
    class GPUWorker;
    class GPUStream;
    class GPUFBMemory;
    class GPUDynamicFBMemory;
    class GPUZCMemory;
    class GPUFBIBMemory;
    class GPU;
    class HipModule;

    extern HipModule *hip_module_singleton;

    // an interface for receiving completion notification for a GPU operation
    //  (right now, just copies)
    class GPUCompletionNotification {
    public:
      virtual ~GPUCompletionNotification(void) {}

      virtual void request_completed(void) = 0;
    };

    class GPUPreemptionWaiter : public GPUCompletionNotification {
    public:
      GPUPreemptionWaiter(GPU *gpu);
      virtual ~GPUPreemptionWaiter(void) {}

    public:
      virtual void request_completed(void);

    public:
      void preempt(void);

    private:
      GPU *const gpu;
      Event wait_event;
    };

    class GPUWorkFence : public Realm::Operation::AsyncWorkItem {
    public:
      GPUWorkFence(Realm::Operation *op);

      virtual void mark_finished(bool successful);

      virtual void request_cancellation(void);

      void enqueue_on_stream(GPUStream *stream);

      virtual void print(std::ostream &os) const;

      IntrusiveListLink<GPUWorkFence> fence_list_link;
      REALM_PMTA_DEFN(GPUWorkFence, IntrusiveListLink<GPUWorkFence>, fence_list_link);
      typedef IntrusiveList<GPUWorkFence, REALM_PMTA_USE(GPUWorkFence, fence_list_link),
                            DummyLock>
          FenceList;

    protected:
      static void cuda_callback(hipStream_t stream, hipError_t res, void *data);
    };

    class GPUWorkStart : public Realm::Operation::AsyncWorkItem {
    public:
      GPUWorkStart(Realm::Operation *op);

      virtual void request_cancellation(void) { return; };

      void enqueue_on_stream(GPUStream *stream);

      virtual void print(std::ostream &os) const;

      void mark_gpu_work_start();

    protected:
      static void cuda_start_callback(hipStream_t stream, hipError_t res, void *data);
    };

    // a class that represents a HIP stream and work associated with
    //  it (e.g. queued copies, events in flight)
    // a stream is also associated with a GPUWorker that it will register
    //  with when async work needs doing
    class GPUStream {
    public:
      GPUStream(GPU *_gpu, GPUWorker *_worker, int rel_priority = 0);
      ~GPUStream(void);

      GPU *get_gpu(void) const;
      REALM_INTERNAL_API_EXTERNAL_LINKAGE hipStream_t
      get_stream(void) const; // needed by librealm_kokkos.so

      // may be called by anybody to enqueue a copy or an event
      void add_fence(GPUWorkFence *fence);
      void add_start_event(GPUWorkStart *start);
      void add_notification(GPUCompletionNotification *notification);
      void wait_on_streams(const std::set<GPUStream *> &other_streams);

      // atomically checks rate limit counters and returns true if 'bytes'
      //  worth of copies can be submitted or false if not (in which case
      //  the progress counter on the xd will be updated when it should try
      //  again)
      bool ok_to_submit_copy(size_t bytes, XferDes *xd);

      // to be called by a worker (that should already have the GPU context
      //   current) - returns true if any work remains
      bool reap_events(TimeLimit work_until);

    protected:
      // may only be tested with lock held
      bool has_work(void) const;

      void add_event(hipEvent_t event, GPUWorkFence *fence,
                     GPUCompletionNotification *notification = NULL,
                     GPUWorkStart *start = NULL);

      GPU *gpu;
      GPUWorker *worker;

      hipStream_t stream;

      Mutex mutex;

      struct PendingEvent {
        hipEvent_t event;
        GPUWorkFence *fence;
        GPUWorkStart *start;
        GPUCompletionNotification *notification;
      };
#ifdef USE_CQ
      Realm::CircularQueue<PendingEvent> pending_events;
#else
      std::deque<PendingEvent> pending_events;
#endif
    };

    // a GPUWorker is responsible for making progress on one or more GPUStreams -
    //  this may be done directly by a GPUProcessor or in a background thread
    //  spawned for the purpose
    class GPUWorker : public BackgroundWorkItem {
    public:
      GPUWorker(void);
      virtual ~GPUWorker(void);

      // adds a stream that has work to be done
      void add_stream(GPUStream *s);

      // used to start a dedicate thread (mutually exclusive with being
      //  registered with a background work manager)
      void start_background_thread(Realm::CoreReservationSet &crs, size_t stack_size);
      void shutdown_background_thread(void);

      bool do_work(TimeLimit work_until);

    public:
      void thread_main(void);

    protected:
      // used by the background thread
      // processes work on streams, optionally sleeping for work to show up
      // returns true if work remains to be done
      bool process_streams(bool sleep_on_empty);

      Mutex lock;
      Mutex::CondVar condvar;

      typedef CircularQueue<GPUStream *, 16> ActiveStreamQueue;
      ActiveStreamQueue active_streams;

      // used by the background thread (if any)
      Realm::CoreReservation *core_rsrv;
      Realm::Thread *worker_thread;
      bool thread_sleeping;
      atomic<bool> worker_shutdown_requested;
    };

    // a little helper class to manage a pool of CUevents that can be reused
    //  to reduce alloc/destroy overheads
    class GPUEventPool {
    public:
      GPUEventPool(int _batch_size = 256);

      // allocating the initial batch of events and cleaning up are done with
      //  these methods instead of constructor/destructor because we don't
      //  manage the GPU context in this helper class
      void init_pool(int init_size = 0 /* default == batch size */);
      void empty_pool(void);

      hipEvent_t get_event(bool external = false);
      void return_event(hipEvent_t e, bool external = false);

    protected:
      Mutex mutex;
      int batch_size, current_size, total_size, external_count;
      std::vector<hipEvent_t> available_events;
    };

    // when the runtime hijack is not enabled/active, a cuCtxSynchronize
    //  is required to ensure a task's completion event covers all of its
    //  actions - rather than blocking an important thread, we create a
    //  small thread pool to handle these
    class ContextSynchronizer {
    public:
      ContextSynchronizer(GPU *_gpu, int _device_id, CoreReservationSet &crs,
                          int _max_threads);
      ~ContextSynchronizer();

      void add_fence(GPUWorkFence *fence);

      void shutdown_threads();

      void thread_main();

    protected:
      GPU *gpu;
      // hipCtx_t context;
      int device_id;
      int max_threads;
      Mutex mutex;
      Mutex::CondVar condvar;
      bool shutdown_flag;
      GPUWorkFence::FenceList fences;
      int total_threads, sleeping_threads, syncing_threads;
      std::vector<Thread *> worker_threads;
      CoreReservation *core_rsrv;
    };

    // a GPU object represents our use of a given HIP-capable GPU - this will
    //  have an associated HIP context, a (possibly shared) worker thread, a
    //  processor, and an FB memory (the ZC memory is shared across all GPUs)
    class GPU {
    public:
      GPU(HipModule *_module, GPUInfo *_info, GPUWorker *worker, int _device_id);
      ~GPU(void);

      void push_context(void);
      void pop_context(void);

      void create_processor(RuntimeImpl *runtime, size_t stack_size);
      void create_fb_memory(RuntimeImpl *runtime, size_t size, size_t ib_size);
      void create_dynamic_fb_memory(RuntimeImpl *runtime, size_t max_size);

      void create_dma_channels(Realm::RuntimeImpl *r);

      bool can_access_peer(const GPU *peer) const;

      GPUStream *find_stream(hipStream_t stream) const;
      REALM_INTERNAL_API_EXTERNAL_LINKAGE GPUStream *
      get_null_task_stream(void) const; // needed by librealm_kokkos.so
      GPUStream *get_next_task_stream(bool create = false);
      GPUStream *get_next_d2d_stream();
      void launch_batch_affine_kernel(void *copy_info, size_t dim, size_t elemSize,
                                      size_t volume, GPUStream *stream, size_t arg_size);
      void launch_batch_affine_fill_kernel(void *fill_info, size_t dim, size_t elem_size,
                                           size_t volume, size_t arg_size,
                                           GPUStream *stream);
      void launch_transpose_kernel(MemcpyTransposeInfo<size_t> &copy_info,
                                   size_t elemSize, GPUStream *stream);

      void launch_indirect_copy_kernel(void *copy_info, size_t dim, size_t addr_size,
                                       size_t field_size, size_t volume, size_t arg_size,
                                       GPUStream *stream);
      bool is_accessible_host_mem(const MemoryImpl *mem) const;
      bool is_accessible_gpu_mem(const MemoryImpl *mem) const;

    protected:
      hipModule_t load_hip_module(const void *data);

    public:
      HipModule *module = nullptr;
      GPUInfo *info = nullptr;
      GPUWorker *worker = nullptr;
      GPUProcessor *proc = nullptr;
      GPUFBMemory *fbmem = nullptr;
      GPUDynamicFBMemory *fb_dmem = nullptr;
      GPUFBIBMemory *fb_ibmem = nullptr;

      // hipCtx_t context;
      int device_id = -1;
      hipModule_t device_module = nullptr;

      struct GPUFuncInfo {
        hipFunction_t func;
        int occ_num_threads;
        int occ_num_blocks;
      };

      // The maximum value of log2(type_bytes) that hip kernels handle.
      // log2(1 byte)   --> 0
      // log2(2 bytes)  --> 1
      // log2(4 bytes)  --> 2
      // log2(8 bytes)  --> 3
      // log2(16 bytes) --> 4
      static const size_t HIP_MEMCPY_KERNEL_MAX2_LOG2_BYTES = 5;

      GPUFuncInfo batch_affine_kernels[REALM_MAX_DIM][HIP_MEMCPY_KERNEL_MAX2_LOG2_BYTES];
      GPUFuncInfo batch_affine_fill_kernels[REALM_MAX_DIM]
                                           [HIP_MEMCPY_KERNEL_MAX2_LOG2_BYTES];
      GPUFuncInfo fill_affine_large_kernels[REALM_MAX_DIM]
                                           [HIP_MEMCPY_KERNEL_MAX2_LOG2_BYTES];
      GPUFuncInfo indirect_copy_kernels[REALM_MAX_DIM][HIP_MEMCPY_KERNEL_MAX2_LOG2_BYTES]
                                       [HIP_MEMCPY_KERNEL_MAX2_LOG2_BYTES];
      GPUFuncInfo transpose_kernels[HIP_MEMCPY_KERNEL_MAX2_LOG2_BYTES];

      char *fbmem_base = nullptr;

      char *fb_ibmem_base = nullptr;

      // which system memories have been registered and can be used for cuMemcpyAsync
      std::set<Memory> pinned_sysmems;

      // managed memories we can concurrently access
      std::set<Memory> managed_mems;

      // which other FBs we have peer access to
      std::set<Memory> peer_fbs;

      // streams for different copy types and a pile for actual tasks
      GPUStream *host_to_device_stream = nullptr;
      GPUStream *device_to_host_stream = nullptr;
      GPUStream *device_to_device_stream = nullptr;
      std::vector<GPUStream *> device_to_device_streams;
      std::vector<GPUStream *> peer_to_peer_streams; // indexed by target
      std::vector<GPUStream *> task_streams;
      atomic<unsigned> next_task_stream = atomic<unsigned>(0);
      atomic<unsigned> next_d2d_stream = atomic<unsigned>(0);

      GPUEventPool event_pool;

      // this can technically be different in each context (but probably isn't
      //  in practice)
      int least_stream_priority, greatest_stream_priority;

      struct HipIpcMapping {
        NodeID owner;
        Memory mem;
        uintptr_t local_base;
        uintptr_t address_offset; // add to convert from original to local base
      };
      std::vector<HipIpcMapping> hipipc_mappings;
      std::map<NodeID, GPUStream *> hipipc_streams;

      const HipIpcMapping *find_ipc_mapping(Memory mem) const;
    };

    // helper to push/pop a GPU's context by scope
    class AutoGPUContext {
    public:
      AutoGPUContext(GPU &_gpu);
      AutoGPUContext(GPU *_gpu);
      ~AutoGPUContext(void);

    protected:
      GPU *gpu;
    };

    class REALM_INTERNAL_API_EXTERNAL_LINKAGE GPUProcessor // needed by librealm_kokkos.so
      : public Realm::LocalTaskProcessor {
    public:
      GPUProcessor(RuntimeImpl *runtime_impl, GPU *_gpu, Processor _me,
                   Realm::CoreReservationSet &crs, size_t _stack_size);
      virtual ~GPUProcessor(void);

    public:
      virtual bool register_task(Processor::TaskFuncID func_id, CodeDescriptor &codedesc,
                                 const ByteArrayRef &user_data);

      virtual void shutdown(void);

    protected:
      virtual void execute_task(Processor::TaskFuncID func_id,
                                const ByteArrayRef &task_args);

    public:
      static GPUProcessor *get_current_gpu_proc(void);

      void stream_wait_on_event(hipStream_t stream, hipEvent_t event);
      void stream_synchronize(hipStream_t stream);
      void device_synchronize(void);

      void gpu_memcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind);
      void gpu_memcpy_async(void *dst, const void *src, size_t size, hipMemcpyKind kind,
                            hipStream_t stream);
      void gpu_memset(void *dst, int value, size_t count);
      void gpu_memset_async(void *dst, int value, size_t count, hipStream_t stream);

    public:
      GPU *gpu;

      bool block_on_synchronize;
      ContextSynchronizer ctxsync;

    protected:
      Realm::CoreReservation *core_rsrv;

      struct GPUTaskTableEntry {
        Processor::TaskFuncPtr fnptr;
        Hip::StreamAwareTaskFuncPtr stream_aware_fnptr;
        ByteArray user_data;
      };

      // we're not using the parent's task table, but we can use the mutex
      // RWLock task_table_mutex;
      std::map<Processor::TaskFuncID, GPUTaskTableEntry> gpu_task_table;
    };

    // this can be attached to any MemoryImpl if the underlying memory is
    //  guaranteed to belong to a given device - this will allow that
    //  context's processor and dma channels to work with it
    // the creator is expected to know what device they want but need
    //  not know which GPU object that corresponds to
    class HipDeviceMemoryInfo : public ModuleSpecificInfo {
    public:
      HipDeviceMemoryInfo(int _device_id);

      int device_id;
      GPU *gpu;
    };

    class GPUFBMemory : public LocalManagedMemory {
    public:
      GPUFBMemory(RuntimeImpl *_runtime_impl, Memory _me, GPU *_gpu, char *_base,
                  size_t _size);

      virtual ~GPUFBMemory(void);

      // these work, but they are SLOW
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      // GPUFBMemory supports ExternalHipMemoryResource and
      //  ExternalHipArrayResource (not implemented)
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t &inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

      // for re-registration purposes, generate an ExternalInstanceResource *
      //  (if possible) for a given instance, or a subset of one
      virtual ExternalInstanceResource *
      generate_resource_info(RegionInstanceImpl *inst, const IndexSpaceGeneric *subspace,
                             span<const FieldID> fields, bool read_only);

    public:
      GPU *gpu;
      char *base;
      NetworkSegment local_segment;
    };

    class GPUDynamicFBMemory : public MemoryImpl {
    public:
      GPUDynamicFBMemory(RuntimeImpl *_runtime_impl, Memory _me, GPU *_gpu,
                         size_t _max_size);

      virtual ~GPUDynamicFBMemory(void);
      void cleanup(void);

      // deferred allocation not supported
      virtual AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
                                                          bool need_alloc_result,
                                                          bool poisoned,
                                                          TimeLimit work_until);

      virtual void release_storage_immediate(RegionInstanceImpl *inst, bool poisoned,
                                             TimeLimit work_until);

      // these work, but they are SLOW
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      // GPUDynamicFBMemory supports ExternalHipMemoryResource and
      //  ExternalHipArrayResource (not implemented)
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t &inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

      // for re-registration purposes, generate an ExternalInstanceResource *
      //  (if possible) for a given instance, or a subset of one
      virtual ExternalInstanceResource *
      generate_resource_info(RegionInstanceImpl *inst, const IndexSpaceGeneric *subspace,
                             span<const FieldID> fields, bool read_only);

    public:
      GPU *gpu;
      Mutex mutex;
      size_t cur_size;
      std::map<RegionInstance, std::pair<void *, size_t>> alloc_bases;
    };

    class GPUZCMemory : public LocalManagedMemory {
    public:
      GPUZCMemory(RuntimeImpl *_runtime_impl, Memory _me, char *_gpu_base,
                  void *_cpu_base, size_t _size, MemoryKind _kind,
                  Memory::Kind _lowlevel_kind);

      virtual ~GPUZCMemory(void);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      // GPUZCMemory supports ExternalHipPinnedHostResource
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t &inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

      // for re-registration purposes, generate an ExternalInstanceResource *
      //  (if possible) for a given instance, or a subset of one
      virtual ExternalInstanceResource *
      generate_resource_info(RegionInstanceImpl *inst, const IndexSpaceGeneric *subspace,
                             span<const FieldID> fields, bool read_only);

    public:
      char *gpu_base;
      char *cpu_base;
      NetworkSegment local_segment;
    };

    class GPUFBIBMemory : public IBMemory {
    public:
      GPUFBIBMemory(RuntimeImpl *_runtime_impl, Memory _me, GPU *_gpu, char *_base,
                    size_t _size);

    public:
      GPU *gpu;
      char *base;
      NetworkSegment local_segment;
    };

    class GPURequest;

    class GPUCompletionEvent : public GPUCompletionNotification {
    public:
      void request_completed(void);

      GPURequest *req;
    };

    class GPURequest : public Request {
    public:
      const void *src_base;
      void *dst_base;
      // off_t src_gpu_off, dst_gpu_off;
      GPU *dst_gpu;
      GPUCompletionEvent event;
    };

    class GPUIndirectTransferCompletion : public GPUCompletionNotification {
    public:
      GPUIndirectTransferCompletion(
          XferDes *_xd, int _read_port_idx, size_t _read_offset, size_t _read_size,
          int _write_port_idx, size_t _write_offset, size_t _write_size,
          int _read_ind_port_idx = -1, size_t _read_ind_offset = 0,
          size_t _read_ind_size = 0, int _write_ind_port_idx = -1,
          size_t _write_ind_offset = 0, size_t _write_ind_size = 0);

      virtual void request_completed(void);

    protected:
      XferDes *xd;
      int read_port_idx;
      size_t read_offset, read_size;
      int read_ind_port_idx;
      size_t read_ind_offset, read_ind_size;
      int write_port_idx;
      size_t write_offset, write_size;
      int write_ind_port_idx;
      size_t write_ind_offset, write_ind_size;
    };

    class GPUTransferCompletion : public GPUCompletionNotification {
    public:
      GPUTransferCompletion(XferDes *_xd, int _read_port_idx, size_t _read_offset,
                            size_t _read_size, int _write_port_idx, size_t _write_offset,
                            size_t _write_size);

      virtual void request_completed(void);

    protected:
      XferDes *xd;
      int read_port_idx;
      size_t read_offset, read_size;
      int write_port_idx;
      size_t write_offset, write_size;
    };
    class MemSpecificHipArray : public MemSpecificInfo {
    public:
      MemSpecificHipArray(hipArray_t _array);
      virtual ~MemSpecificHipArray();

      hipArray_t array;
    };

    class AddressInfoHipArray : public TransferIterator::AddressInfoCustom {
    public:
      virtual int set_rect(const RegionInstanceImpl *inst,
                           const InstanceLayoutPieceBase *piece, size_t field_size,
                           size_t field_offset, int ndims, const int64_t lo[/*ndims*/],
                           const int64_t hi[/*ndims*/], const int order[/*ndims*/]);

      hipArray_t array;
      int dim;
      size_t pos[3];
      size_t width_in_bytes, height, depth;
    };

    class GPUChannel;

    class GPUXferDes : public XferDes {
    public:
      GPUXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                 XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                 const std::vector<XferDesPortInfo> &outputs_info, int _priority);

      long get_requests(Request **requests, long nr);

      bool progress_xd(GPUChannel *channel, TimeLimit work_until);

    private:
      std::vector<GPU *> src_gpus, dst_gpus;
      std::vector<bool> dst_is_ipc;
    };

    class GPUIndirectChannel;

    class GPUIndirectXferDes : public XferDes {
    public:
      GPUIndirectXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                         XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                         const std::vector<XferDesPortInfo> &outputs_info, int _priority,
                         XferDesRedopInfo _redop_info);

      long get_requests(Request **requests, long nr);
      bool progress_xd(GPUIndirectChannel *channel, TimeLimit work_until);

    protected:
      std::vector<GPU *> src_gpus, dst_gpus;
      std::vector<bool> dst_is_ipc;
    };

    class GPUIndirectChannel
      : public SingleXDQChannel<GPUIndirectChannel, GPUIndirectXferDes> {
    public:
      GPUIndirectChannel(GPU *_src_gpu, XferDesKind _kind, BackgroundWorkManager *bgwork);
      ~GPUIndirectChannel();

      // multi-threading of copies for a given device is disabled by
      static const bool is_ordered = true;

      virtual bool needs_wrapping_iterator() const;
      virtual Memory suggest_ib_memories() const;

      virtual RemoteChannelInfo *construct_remote_info() const;

      virtual uint64_t
      supports_path(ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
                    CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
                    size_t total_bytes, const std::vector<size_t> *src_frags,
                    const std::vector<size_t> *dst_frags, XferDesKind *kind_ret = 0,
                    unsigned *bw_ret = 0, unsigned *lat_ret = 0);

      virtual bool supports_indirection_memory(Memory mem) const;

      virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                       XferDesID guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int priority, XferDesRedopInfo redop_info,
                                       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request **requests, long nr);
      GPU *get_gpu() const { return src_gpu; }

    protected:
      friend class GPUIndirectXferDes;
      GPU *src_gpu;
    };

    class GPUIndirectRemoteChannelInfo : public SimpleRemoteChannelInfo {
    public:
      GPUIndirectRemoteChannelInfo(NodeID _owner, XferDesKind _kind,
                                   uintptr_t _remote_ptr,
                                   const std::vector<Channel::SupportedPath> &_paths,
                                   const std::vector<Memory> &_indirect_memories);

      virtual RemoteChannel *create_remote_channel();

      template <typename S>
      bool serialize(S &serializer) const;

      template <typename S>
      static RemoteChannelInfo *deserialize_new(S &deserializer);

    protected:
      static Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                      GPUIndirectRemoteChannelInfo>
          serdez_subclass;
    };

    class GPUIndirectRemoteChannel : public RemoteChannel {
      friend class GPUIndirectRemoteChannelInfo;

    public:
      GPUIndirectRemoteChannel(uintptr_t _remote_ptr,
                               const std::vector<Memory> &_indirect_memories);
      virtual Memory suggest_ib_memories() const;
      virtual bool needs_wrapping_iterator() const;
      virtual uint64_t
      supports_path(ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
                    CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
                    size_t total_bytes, const std::vector<size_t> *src_frags,
                    const std::vector<size_t> *dst_frags, XferDesKind *kind_ret /*= 0*/,
                    unsigned *bw_ret /*= 0*/, unsigned *lat_ret /*= 0*/);
    };

    class GPUChannel : public SingleXDQChannel<GPUChannel, GPUXferDes> {
    public:
      GPUChannel(GPU *_src_gpu, XferDesKind _kind, BackgroundWorkManager *bgwork);
      ~GPUChannel();

      // multi-threading of cuda copies for a given device is disabled by
      //  default (can be re-enabled with -cuda:mtdma 1)
      static const bool is_ordered = true;

      virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                       XferDesID guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int priority, XferDesRedopInfo redop_info,
                                       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request **requests, long nr);
      GPU *get_gpu() const { return src_gpu; }

    private:
      GPU *src_gpu;
      // std::deque<Request*> pending_copies;
    };

    class GPUfillChannel;

    class GPUfillXferDes : public XferDes {
    public:
      GPUfillXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                     XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                     const std::vector<XferDesPortInfo> &outputs_info, int _priority,
                     const void *_fill_data, size_t _fill_size, size_t _fill_total);

      long get_requests(Request **requests, long nr);

      bool progress_xd(GPUfillChannel *channel, TimeLimit work_until);

    protected:
      size_t reduced_fill_size;
    };

    class GPUfillChannel : public SingleXDQChannel<GPUfillChannel, GPUfillXferDes> {
    public:
      GPUfillChannel(GPU *_gpu, BackgroundWorkManager *bgwork);

      // multiple concurrent cuda fills ok
      static const bool is_ordered = false;

      virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                       XferDesID guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int priority, XferDesRedopInfo redop_info,
                                       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request **requests, long nr);

    protected:
      friend class GPUfillXferDes;

      GPU *gpu;
    };

    class GPUreduceChannel;
    class GPUreduceXferDes : public XferDes {
    public:
      GPUreduceXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                       XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                       const std::vector<XferDesPortInfo> &outputs_info, int _priority,
                       XferDesRedopInfo _redop_info);

      long get_requests(Request **requests, long nr);

      bool progress_xd(GPUreduceChannel *channel, TimeLimit work_until);
      bool fast_reduction_kernel_mode(GPUreduceChannel *channel, size_t max_bytes,
                                      XferPort *in_port, XferPort *out_port,
                                      size_t in_span_start, size_t out_span_start);
      void *describe_kernel_variant(bool is_advanced);
      void setup_redop_kernel(GPUreduceChannel *channel, void *redop_args,
                              const size_t in_span_start, const size_t out_span_start,
                              const size_t in_elem_size, const size_t out_elem_size,
                              const size_t elems, const bool has_transpose);

    protected:
      XferDesRedopInfo redop_info;
      const ReductionOpUntyped *redop;
      void *kernel_host_proxy;
      void *kernel_host_proxy_advanced;
      void *kernel_host_proxy_transpose;
      GPUStream *stream;
      std::vector<GPU *> src_gpus;
      std::vector<bool> src_is_ipc;
    };

    class GPUreduceChannel : public SingleXDQChannel<GPUreduceChannel, GPUreduceXferDes> {
    public:
      GPUreduceChannel(GPU *_gpu, BackgroundWorkManager *bgwork);

      // multiple concurrent cuda reduces ok
      static const bool is_ordered = false;

      virtual bool supports_redop(ReductionOpID redop_id) const;

      virtual RemoteChannelInfo *construct_remote_info() const;

      virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                       XferDesID guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int priority, XferDesRedopInfo redop_info,
                                       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request **requests, long nr);

    protected:
      friend class GPUreduceXferDes;

      GPU *gpu;
    };

    class GPUreduceRemoteChannelInfo : public SimpleRemoteChannelInfo {
    public:
      GPUreduceRemoteChannelInfo(NodeID _owner, XferDesKind _kind, uintptr_t _remote_ptr,
                                 const std::vector<Channel::SupportedPath> &_paths);

      virtual RemoteChannel *create_remote_channel();

      template <typename S>
      bool serialize(S &serializer) const;

      template <typename S>
      static RemoteChannelInfo *deserialize_new(S &deserializer);

    protected:
      static Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                      GPUreduceRemoteChannelInfo>
          serdez_subclass;
    };

    class GPUreduceRemoteChannel : public RemoteChannel {
      friend class GPUreduceRemoteChannelInfo;

      GPUreduceRemoteChannel(uintptr_t _remote_ptr);
    };

    // active messages for establishing cuda ipc mappings

    struct HipIpcRequest {
#ifdef REALM_ON_LINUX
      long hostid; // POSIX hostid
#endif

      static void handle_message(NodeID sender, const HipIpcRequest &args,
                                 const void *data, size_t datalen);
    };

    struct HipIpcResponse {
      unsigned count;

      static void handle_message(NodeID sender, const HipIpcResponse &args,
                                 const void *data, size_t datalen);
    };

    struct HipIpcRelease {

      static void handle_message(NodeID sender, const HipIpcRelease &args,
                                 const void *data, size_t datalen);
    };

    class GPUReplHeapListener : public ReplicatedHeap::Listener {
    public:
      GPUReplHeapListener(HipModule *_module);

      virtual void chunk_created(void *base, size_t bytes);
      virtual void chunk_destroyed(void *base, size_t bytes);

    protected:
      HipModule *module;
    };

  }; // namespace Hip

}; // namespace Realm

#endif
