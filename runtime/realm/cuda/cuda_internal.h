/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#ifndef REALM_CUDA_INTERNAL_H
#define REALM_CUDA_INTERNAL_H

#include "realm/cuda/cuda_module.h"

#include <memory>
#include <cuda.h>
#include <nvml.h>
#if defined(REALM_USE_CUDART_HIJACK)
#include <cuda_runtime_api.h>   // For cudaDeviceProp
#endif

// For CUDA runtime's dim3 definition
#include <vector_types.h>

#if defined(REALM_CUDA_DYNAMIC_LOAD) && (CUDA_VERSION >= 11030)
#include <cudaTypedefs.h>
#endif

#include "realm/operation.h"
#include "realm/threads.h"
#include "realm/circ_queue.h"
#include "realm/indexspace.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/bgwork.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"
#include "realm/cuda/cuda_memcpy.h"

#if CUDART_VERSION < 11000
#define CHECK_CUDART(cmd)                                                      \
  do {                                                                         \
    int ret = (int)(cmd);                                                      \
    if (ret != 0) {                                                            \
      fprintf(stderr, "CUDART: %s = %d\n", #cmd, ret);                         \
      assert(0);                                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#else
// Since CUDA TK11.0, runtime and driver error codes are 1:1 correlated
#define CHECK_CUDART(cmd) CHECK_CU((CUresult)(cmd))
#endif

// Need CUDA 6.5 or later for good error reporting
#if CUDA_VERSION >= 6050
#define REPORT_CU_ERROR(level, cmd, ret)                                                 \
  do {                                                                                   \
    const char *name, *str;                                                              \
    CUDA_DRIVER_FNPTR(cuGetErrorName)(ret, &name);                                       \
    CUDA_DRIVER_FNPTR(cuGetErrorString)(ret, &str);                                      \
    log_gpu.newmsg(level) << __FILE__ << '(' << __LINE__ << "):" << cmd << " = " << ret  \
                          << '(' << name << "): " << str;                                \
  } while(0)
#else
#define REPORT_CU_ERROR(level, cmd, ret)                                                 \
  do {                                                                                   \
    log_gpu.newmsg(level) << __FILE__ << '(' << __LINE__ << "):" << cmd << " = " << ret  \
  } while(0)
#endif

#define CHECK_CU(cmd)                                                                    \
  do {                                                                                   \
    CUresult ret = (cmd);                                                                \
    if(ret != CUDA_SUCCESS) {                                                            \
      REPORT_CU_ERROR(Logger::LEVEL_ERROR, #cmd, ret);                                   \
      abort();                                                                           \
    }                                                                                    \
  } while(0)

#define REPORT_NVML_ERROR(level, cmd, ret)                                               \
  do {                                                                                   \
    log_gpu.newmsg(level) << __FILE__ << '(' << __LINE__ << "):" << cmd << " = " << ret; \
  } while(0)

#define CHECK_NVML(cmd)                                                                  \
  do {                                                                                   \
    nvmlReturn_t ret = (cmd);                                                            \
    if(ret != NVML_SUCCESS) {                                                            \
      REPORT_NVML_ERROR(Logger::LEVEL_ERROR, #cmd, ret);                                 \
      abort();                                                                           \
    }                                                                                    \
  } while(0)

namespace Realm {

  namespace Cuda {

    struct GPUInfo {
      int index; // index used by CUDA runtime
      CUdevice device;
      nvmlDevice_t nvml_dev;
      CUuuid uuid;
      int major;
      int minor;
      static const size_t MAX_NAME_LEN = 256;
      char name[MAX_NAME_LEN];
      size_t totalGlobalMem;
      static const size_t MAX_NUMA_NODE_LEN = 20;
      bool has_numa_preference;
      unsigned long numa_node_affinity[MAX_NUMA_NODE_LEN];
      std::set<CUdevice> peers; // other GPUs we can do p2p copies with
      int pci_busid;
      int pci_domainid;
      int pci_deviceid;
      size_t pci_bandwidth; // Current enabled pci-e bandwidth
      bool host_gpu_same_va = false;
      std::vector<size_t> logical_peer_bandwidth;
      std::vector<size_t> logical_peer_latency;

#ifdef REALM_USE_CUDART_HIJACK
      cudaDeviceProp prop;
#endif
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
    class GPUAllocation;
    class GPU;
    class CudaModule;

    extern CudaModule *cuda_module_singleton;

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

    // An abstract base class for all GPU memcpy operations
    class GPUMemcpy { //: public GPUJob {
    public:
      GPUMemcpy(GPU *_gpu, GPUMemcpyKind _kind);
      virtual ~GPUMemcpy(void) {}

    public:
      virtual void execute(GPUStream *stream) = 0;

    public:
      GPU *const gpu;

    protected:
      GPUMemcpyKind kind;
    };

    class GPUWorkFence : public Realm::Operation::AsyncWorkItem {
    public:
      GPUWorkFence(Realm::Operation *op);

      virtual void request_cancellation(void);

      void enqueue_on_stream(GPUStream *stream);

      virtual void print(std::ostream &os) const;

      IntrusiveListLink<GPUWorkFence> fence_list_link;
      REALM_PMTA_DEFN(GPUWorkFence, IntrusiveListLink<GPUWorkFence>, fence_list_link);
      typedef IntrusiveList<GPUWorkFence, REALM_PMTA_USE(GPUWorkFence, fence_list_link),
                            DummyLock>
          FenceList;

    protected:
      static void cuda_callback(CUstream stream, CUresult res, void *data);
    };

    class GPUWorkStart : public Realm::Operation::AsyncWorkItem {
    public:
      GPUWorkStart(Realm::Operation *op);

      virtual void request_cancellation(void) { return; };

      void enqueue_on_stream(GPUStream *stream);

      virtual void print(std::ostream &os) const;

      void mark_gpu_work_start();

    protected:
      static void cuda_start_callback(CUstream stream, CUresult res, void *data);
    };

    class GPUMemcpyFence : public GPUMemcpy {
    public:
      GPUMemcpyFence(GPU *_gpu, GPUMemcpyKind _kind, GPUWorkFence *_fence);

      virtual void execute(GPUStream *stream);

    protected:
      GPUWorkFence *fence;
    };

    class GPUMemcpy1D : public GPUMemcpy {
    public:
      GPUMemcpy1D(GPU *_gpu, void *_dst, const void *_src, size_t _bytes,
                  GPUMemcpyKind _kind, GPUCompletionNotification *_notification);

      virtual ~GPUMemcpy1D(void);

    public:
      void do_span(off_t pos, size_t len);
      virtual void execute(GPUStream *stream);

    protected:
      void *dst;
      const void *src;
      size_t elmt_size;
      GPUCompletionNotification *notification;

    private:
      GPUStream *local_stream; // used by do_span
    };

    class GPUMemcpy2D : public GPUMemcpy {
    public:
      GPUMemcpy2D(GPU *_gpu, void *_dst, const void *_src, off_t _dst_stride,
                  off_t _src_stride, size_t _bytes, size_t _lines, GPUMemcpyKind _kind,
                  GPUCompletionNotification *_notification);

      virtual ~GPUMemcpy2D(void);

    public:
      virtual void execute(GPUStream *stream);

    protected:
      void *dst;
      const void *src;
      off_t dst_stride, src_stride;
      size_t bytes, lines;
      GPUCompletionNotification *notification;
    };

    class GPUMemcpy3D : public GPUMemcpy {
    public:
      GPUMemcpy3D(GPU *_gpu, void *_dst, const void *_src, off_t _dst_stride,
                  off_t _src_stride, off_t _dst_pstride, off_t _src_pstride,
                  size_t _bytes, size_t _height, size_t _depth, GPUMemcpyKind _kind,
                  GPUCompletionNotification *_notification);

      virtual ~GPUMemcpy3D(void);

    public:
      virtual void execute(GPUStream *stream);

    protected:
      void *dst;
      const void *src;
      off_t dst_stride, src_stride, dst_pstride, src_pstride;
      size_t bytes, height, depth;
      GPUCompletionNotification *notification;
    };

    class GPUMemset1D : public GPUMemcpy {
    public:
      GPUMemset1D(GPU *_gpu, void *_dst, size_t _bytes, const void *_fill_data,
                  size_t _fill_data_size, GPUCompletionNotification *_notification);

      virtual ~GPUMemset1D(void);

    public:
      virtual void execute(GPUStream *stream);

    protected:
      void *dst;
      size_t bytes;
      static const size_t MAX_DIRECT_SIZE = 8;
      union {
        char direct[8];
        char *indirect;
      } fill_data;
      size_t fill_data_size;
      GPUCompletionNotification *notification;
    };

    class GPUMemset2D : public GPUMemcpy {
    public:
      GPUMemset2D(GPU *_gpu, void *_dst, size_t _dst_stride, size_t _bytes, size_t _lines,
                  const void *_fill_data, size_t _fill_data_size,
                  GPUCompletionNotification *_notification);

      virtual ~GPUMemset2D(void);

    public:
      void do_span(off_t pos, size_t len);
      virtual void execute(GPUStream *stream);

    protected:
      void *dst;
      size_t dst_stride;
      size_t bytes, lines;
      static const size_t MAX_DIRECT_SIZE = 8;
      union {
        char direct[8];
        char *indirect;
      } fill_data;
      size_t fill_data_size;
      GPUCompletionNotification *notification;
    };

    class GPUMemset3D : public GPUMemcpy {
    public:
      GPUMemset3D(GPU *_gpu, void *_dst, size_t _dst_stride, size_t _dst_pstride,
                  size_t _bytes, size_t _height, size_t _depth, const void *_fill_data,
                  size_t _fill_data_size, GPUCompletionNotification *_notification);

      virtual ~GPUMemset3D(void);

    public:
      void do_span(off_t pos, size_t len);
      virtual void execute(GPUStream *stream);

    protected:
      void *dst;
      size_t dst_stride, dst_pstride;
      size_t bytes, height, depth;
      static const size_t MAX_DIRECT_SIZE = 8;
      union {
        char direct[8];
        char *indirect;
      } fill_data;
      size_t fill_data_size;
      GPUCompletionNotification *notification;
    };

    // a class that represents a CUDA stream and work associated with
    //  it (e.g. queued copies, events in flight)
    // a stream is also associated with a GPUWorker that it will register
    //  with when async work needs doing
    class GPUStream {
    public:
      GPUStream(GPU *_gpu, GPUWorker *_worker, int rel_priority = 0);
      ~GPUStream(void);

      GPU *get_gpu(void) const;
      CUstream get_stream(void) const;

      // may be called by anybody to enqueue a copy or an event
      void add_copy(GPUMemcpy *copy);
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
      bool issue_copies(TimeLimit work_until);
      bool reap_events(TimeLimit work_until);

    protected:
      // may only be tested with lock held
      bool has_work(void) const;

      void add_event(CUevent event, GPUWorkFence *fence,
                     GPUCompletionNotification *notification = NULL,
                     GPUWorkStart *start = NULL);

      GPU *gpu;
      GPUWorker *worker;

      CUstream stream;

      Mutex mutex;

#define USE_CQ
#ifdef USE_CQ
      Realm::CircularQueue<GPUMemcpy *> pending_copies;
#else
      std::deque<GPUMemcpy *> pending_copies;
#endif
      bool issuing_copies;

      struct PendingEvent {
	CUevent event;
	GPUWorkFence *fence;
	GPUWorkStart *start;
	GPUCompletionNotification* notification;
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
      void start_background_thread(Realm::CoreReservationSet& crs,
				   size_t stack_size);
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

      CUevent get_event(bool external = false);
      void return_event(CUevent e, bool external = false);

    protected:
      Mutex mutex;
      int batch_size, current_size, total_size, external_count;
      std::vector<CUevent> available_events;
    };

    // when the runtime hijack is not enabled/active, a cuCtxSynchronize
    //  is required to ensure a task's completion event covers all of its
    //  actions - rather than blocking an important thread, we create a
    //  small thread pool to handle these
    class ContextSynchronizer {
    public:
      ContextSynchronizer(GPU *_gpu, CUcontext _context,
			  CoreReservationSet& crs,
			  int _max_threads);
      ~ContextSynchronizer();

      void add_fence(GPUWorkFence *fence);

      void shutdown_threads();

      void thread_main();

    protected:
      GPU *gpu;
      CUcontext context;
      int max_threads;
      Mutex mutex;
      Mutex::CondVar condvar;
      bool shutdown_flag;
      GPUWorkFence::FenceList fences;
      int total_threads, sleeping_threads, syncing_threads;
      std::vector<Thread *> worker_threads;
      CoreReservation *core_rsrv;
    };

    struct FatBin;
    struct RegisteredVariable;
    struct RegisteredFunction;

    // a GPU object represents our use of a given CUDA-capable GPU - this will
    //  have an associated CUDA context, a (possibly shared) worker thread, a 
    //  processor, and an FB memory (the ZC memory is shared across all GPUs)
    class GPU {
    public:
      GPU(CudaModule *_module, GPUInfo *_info, GPUWorker *worker,
	  CUcontext _context);
      ~GPU(void);

      void push_context(void);
      void pop_context(void);

      GPUAllocation &add_allocation(GPUAllocation &&alloc);

#ifdef REALM_USE_CUDART_HIJACK
      void register_fat_binary(const FatBin *data);
      void register_variable(const RegisteredVariable *var);
      void register_function(const RegisteredFunction *func);

      CUfunction lookup_function(const void *func);
      CUdeviceptr lookup_variable(const void *var);
#endif

      void create_processor(RuntimeImpl *runtime, size_t stack_size);
      void create_fb_memory(RuntimeImpl *runtime, size_t size, size_t ib_size);
      void create_dynamic_fb_memory(RuntimeImpl *runtime, size_t max_size);

      void create_dma_channels(Realm::RuntimeImpl *r);

      // copy and operations are asynchronous - use a fence (of the right type)
      //   after all of your copies, or a completion notification for particular copies
      void copy_to_fb(off_t dst_offset, const void *src, size_t bytes,
		      GPUCompletionNotification *notification = 0);

      void copy_from_fb(void *dst, off_t src_offset, size_t bytes,
			GPUCompletionNotification *notification = 0);

      void copy_within_fb(off_t dst_offset, off_t src_offset,
			  size_t bytes,
			  GPUCompletionNotification *notification = 0);

      void copy_to_fb_2d(off_t dst_offset, const void *src,
                         off_t dst_stride, off_t src_stride,
                         size_t bytes, size_t lines,
			 GPUCompletionNotification *notification = 0);

      void copy_to_fb_3d(off_t dst_offset, const void *src,
                         off_t dst_stride, off_t src_stride,
                         off_t dst_height, off_t src_height,
                         size_t bytes, size_t height, size_t depth,
			 GPUCompletionNotification *notification = 0);

      void copy_from_fb_2d(void *dst, off_t src_offset,
                           off_t dst_stride, off_t src_stride,
                           size_t bytes, size_t lines,
			   GPUCompletionNotification *notification = 0);

      void copy_from_fb_3d(void *dst, off_t src_offset,
                           off_t dst_stride, off_t src_stride,
                           off_t dst_height, off_t src_height,
                           size_t bytes, size_t height, size_t depth,
			   GPUCompletionNotification *notification = 0);

      void copy_within_fb_2d(off_t dst_offset, off_t src_offset,
                             off_t dst_stride, off_t src_stride,
                             size_t bytes, size_t lines,
			     GPUCompletionNotification *notification = 0);

      void copy_within_fb_3d(off_t dst_offset, off_t src_offset,
                             off_t dst_stride, off_t src_stride,
                             off_t dst_height, off_t src_height,
                             size_t bytes, size_t height, size_t depth,
			     GPUCompletionNotification *notification = 0);

      void copy_to_peer(GPU *dst, off_t dst_offset, 
                        off_t src_offset, size_t bytes,
			GPUCompletionNotification *notification = 0);

      void copy_to_peer_2d(GPU *dst, off_t dst_offset, off_t src_offset,
                           off_t dst_stride, off_t src_stride,
                           size_t bytes, size_t lines,
			   GPUCompletionNotification *notification = 0);

      void copy_to_peer_3d(GPU *dst, off_t dst_offset, off_t src_offset,
                           off_t dst_stride, off_t src_stride,
                           off_t dst_height, off_t src_height,
                           size_t bytes, size_t height, size_t depth,
			   GPUCompletionNotification *notification = 0);

      // fill operations are also asynchronous - use fence_within_fb at end
      void fill_within_fb(off_t dst_offset,
			  size_t bytes,
			  const void *fill_data, size_t fill_data_size,
			  GPUCompletionNotification *notification = 0);

      void fill_within_fb_2d(off_t dst_offset, off_t dst_stride,
			     size_t bytes, size_t lines,
			     const void *fill_data, size_t fill_data_size,
			     GPUCompletionNotification *notification = 0);

      void fill_within_fb_3d(off_t dst_offset, off_t dst_stride,
			     off_t dst_height,
			     size_t bytes, size_t height, size_t depth,
			     const void *fill_data, size_t fill_data_size,
			     GPUCompletionNotification *notification = 0);

      void fence_to_fb(Realm::Operation *op);
      void fence_from_fb(Realm::Operation *op);
      void fence_within_fb(Realm::Operation *op);
      void fence_to_peer(Realm::Operation *op, GPU *dst);

      bool can_access_peer(const GPU *peer) const;

      GPUStream *find_stream(CUstream stream) const;
      GPUStream *get_null_task_stream(void) const;
      GPUStream *get_next_task_stream(bool create = false);
      GPUStream *get_next_d2d_stream();

      void launch_batch_affine_kernel(void *copy_info, size_t dim,
                                      size_t elemSize, size_t volume,
                                      GPUStream *stream);
      void launch_transpose_kernel(MemcpyTransposeInfo<size_t> &copy_info,
                                   size_t elemSize, GPUStream *stream);

      void launch_indirect_copy_kernel(void *copy_info, size_t dim, size_t addr_size,
                                       size_t field_size, size_t volume,
                                       GPUStream *stream);

    protected:
      CUmodule load_cuda_module(const void *data);

    public:
      CudaModule *module = nullptr;
      GPUInfo *info = nullptr;
      GPUWorker *worker = nullptr;
      GPUProcessor *proc = nullptr;

      std::map<CUdeviceptr, GPUAllocation> allocations;
      GPUFBMemory *fbmem = nullptr;
      GPUDynamicFBMemory *fb_dmem = nullptr;
      GPUFBIBMemory *fb_ibmem = nullptr;

      CUcontext context = nullptr;

      CUmodule device_module = nullptr;

      struct GPUFuncInfo {
        CUfunction func;
        int occ_num_threads;
        int occ_num_blocks;
      };

      // The maximum value of log2(type_bytes) that cuda kernels handle.
      // log2(1 byte)   --> 0
      // log2(2 bytes)  --> 1
      // log2(4 bytes)  --> 2
      // log2(8 bytes)  --> 3
      // log2(16 bytes) --> 4
      static const size_t CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES = 5;

      GPUFuncInfo indirect_copy_kernels[REALM_MAX_DIM][CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES]
                                       [CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES];
      GPUFuncInfo batch_affine_kernels[REALM_MAX_DIM][CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES];
      GPUFuncInfo batch_fill_affine_kernels[REALM_MAX_DIM]
                                           [CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES];
      GPUFuncInfo fill_affine_large_kernels[REALM_MAX_DIM]
                                           [CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES];
      GPUFuncInfo transpose_kernels[CUDA_MEMCPY_KERNEL_MAX2_LOG2_BYTES];

      CUdeviceptr fbmem_base = 0;

      CUdeviceptr fb_ibmem_base = 0;

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

      struct CudaIpcMapping {
        NodeID owner;
        GPU *src_gpu;
        Memory mem;
        uintptr_t local_base;
        uintptr_t address_offset; // add to convert from original to local base
      };
      std::vector<CudaIpcMapping> cudaipc_mappings;
      std::map<NodeID, GPUStream *> cudaipc_streams;

      const CudaIpcMapping *find_ipc_mapping(Memory mem) const;

#ifdef REALM_USE_CUDART_HIJACK
      std::map<const FatBin *, CUmodule> device_modules;
      std::map<const void *, CUfunction> device_functions;
      std::map<const void *, CUdeviceptr> device_variables;
#endif
    };

    // helper to push/pop a GPU's context by scope
    class AutoGPUContext {
    public:
      AutoGPUContext(GPU& _gpu);
      AutoGPUContext(GPU *_gpu);
      ~AutoGPUContext(void);
    protected:
      GPU *gpu;
    };

    class GPUProcessor : public Realm::LocalTaskProcessor {
    public:
      GPUProcessor(GPU *_gpu, Processor _me, Realm::CoreReservationSet& crs,
                   size_t _stack_size);
      virtual ~GPUProcessor(void);

    public:
      virtual bool register_task(Processor::TaskFuncID func_id,
				 CodeDescriptor& codedesc,
				 const ByteArrayRef& user_data);

      virtual void shutdown(void);

    protected:
      virtual void execute_task(Processor::TaskFuncID func_id,
				const ByteArrayRef& task_args);

    public:
      static GPUProcessor *get_current_gpu_proc(void);

#ifdef REALM_USE_CUDART_HIJACK
      // calls that come from the CUDA runtime API
      void push_call_configuration(dim3 grid_dim, dim3 block_dim,
                                   size_t shared_size, void *stream);
      void pop_call_configuration(dim3 *grid_dim, dim3 *block_dim,
                                  size_t *shared_size, void *stream);
#endif

      void stream_wait_on_event(CUstream stream, CUevent event);
      void stream_synchronize(CUstream stream);
      void device_synchronize(void);

#ifdef REALM_USE_CUDART_HIJACK
      void event_record(CUevent event, CUstream stream);
      
      void configure_call(dim3 grid_dim, dim3 block_dim,
			  size_t shared_memory, CUstream stream);
      void setup_argument(const void *arg, size_t size, size_t offset);
      void launch(const void *func);
      void launch_kernel(const void *func, dim3 grid_dim, dim3 block_dim, 
                         void **args, size_t shared_memory, 
                         CUstream stream, bool cooperative = false);
#endif

      void gpu_memcpy(void *dst, const void *src, size_t size);
      void gpu_memcpy_async(void *dst, const void *src, size_t size,
                            CUstream stream);
#ifdef REALM_USE_CUDART_HIJACK
      void gpu_memcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                        size_t width, size_t height);
      void gpu_memcpy2d_async(void *dst, size_t dpitch, const void *src, 
                              size_t spitch, size_t width, size_t height, 
                              CUstream stream);
      void gpu_memcpy_to_symbol(const void *dst, const void *src, size_t size,
				size_t offset);
      void gpu_memcpy_to_symbol_async(const void *dst, const void *src, size_t size,
				      size_t offset,
				      CUstream stream);
      void gpu_memcpy_from_symbol(void *dst, const void *src, size_t size,
				  size_t offset);
      void gpu_memcpy_from_symbol_async(void *dst, const void *src, size_t size,
					size_t offset,
					CUstream stream);
#endif

      void gpu_memset(void *dst, int value, size_t count);
      void gpu_memset_async(void *dst, int value, size_t count, CUstream stream);
    public:
      GPU *gpu;

      // data needed for kernel launches
      struct LaunchConfig {
        dim3 grid;
        dim3 block;
        size_t shared;
	LaunchConfig(dim3 _grid, dim3 _block, size_t _shared);
      };
      struct CallConfig : public LaunchConfig {
        CUstream stream; 
        CallConfig(dim3 _grid, dim3 _block, size_t _shared, CUstream _stream);
      };
      std::vector<CallConfig> launch_configs;
      std::vector<char> kernel_args;
      std::vector<CallConfig> call_configs;
      bool block_on_synchronize;
      ContextSynchronizer ctxsync;
    protected:
      Realm::CoreReservation *core_rsrv;

      struct GPUTaskTableEntry {
	Processor::TaskFuncPtr fnptr;
	Cuda::StreamAwareTaskFuncPtr stream_aware_fnptr;
	ByteArray user_data;
      };

      // we're not using the parent's task table, but we can use the mutex
      //RWLock task_table_mutex;
      std::map<Processor::TaskFuncID, GPUTaskTableEntry> gpu_task_table;
    };

    // this can be attached to any MemoryImpl if the underlying memory is
    //  guaranteed to belong to a given CUcontext - this will allow that
    //  context's processor and dma channels to work with it
    // the creator is expected to know what CUcontext they want but need
    //  not know which GPU object that corresponds to
    class CudaDeviceMemoryInfo : public ModuleSpecificInfo
    {
    public:
      CudaDeviceMemoryInfo(CUcontext _context);

      CUcontext context;
      GPU *gpu;
    };

    class GPUFBMemory : public LocalManagedMemory {
    public:
      GPUFBMemory(Memory _me, GPU *_gpu, CUdeviceptr _base, size_t _size);

      virtual ~GPUFBMemory(void);

      // these work, but they are SLOW
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      // GPUFBMemory supports ExternalCudaMemoryResource and
      //  ExternalCudaArrayResource
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t& inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

      // for re-registration purposes, generate an ExternalInstanceResource *
      //  (if possible) for a given instance, or a subset of one
      virtual ExternalInstanceResource *generate_resource_info(RegionInstanceImpl *inst,
							       const IndexSpaceGeneric *subspace,
							       span<const FieldID> fields,
							       bool read_only);

    public:
      GPU *gpu;
      CUdeviceptr base;
      NetworkSegment local_segment;
    };

    class GPUDynamicFBMemory : public MemoryImpl {
    public:
      GPUDynamicFBMemory(Memory _me, GPU *_gpu, size_t _max_size);

      virtual ~GPUDynamicFBMemory(void);
      void cleanup(void);

      // deferred allocation not supported
      virtual AllocationResult allocate_storage_immediate(RegionInstanceImpl *inst,
							  bool need_alloc_result,
							  bool poisoned,
							  TimeLimit work_until);

      virtual void release_storage_immediate(RegionInstanceImpl *inst,
					     bool poisoned,
					     TimeLimit work_until);

      // these work, but they are SLOW
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      // GPUDynamicFBMemory supports ExternalCudaMemoryResource and
      //  ExternalCudaArrayResource
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t& inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

      // for re-registration purposes, generate an ExternalInstanceResource *
      //  (if possible) for a given instance, or a subset of one
      virtual ExternalInstanceResource *generate_resource_info(RegionInstanceImpl *inst,
							       const IndexSpaceGeneric *subspace,
							       span<const FieldID> fields,
							       bool read_only);

    public:
      GPU *gpu;
      Mutex mutex;
      size_t cur_size;
      std::map<RegionInstance, std::pair<CUdeviceptr, size_t> > alloc_bases;
      NetworkSegment local_segment;
    };

    class GPUZCMemory : public LocalManagedMemory {
    public:
      GPUZCMemory(Memory _me, CUdeviceptr _gpu_base,
                  void *_cpu_base, size_t _size,
                  MemoryKind _kind, Memory::Kind _lowlevel_kind);

      virtual ~GPUZCMemory(void);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      // GPUZCMemory supports ExternalCudaPinnedHostResource
      virtual bool attempt_register_external_resource(RegionInstanceImpl *inst,
                                                      size_t& inst_offset);
      virtual void unregister_external_resource(RegionInstanceImpl *inst);

      // for re-registration purposes, generate an ExternalInstanceResource *
      //  (if possible) for a given instance, or a subset of one
      virtual ExternalInstanceResource *generate_resource_info(RegionInstanceImpl *inst,
							       const IndexSpaceGeneric *subspace,
							       span<const FieldID> fields,
							       bool read_only);

    public:
      CUdeviceptr gpu_base;
      char *cpu_base;
      NetworkSegment local_segment;
    };

    class GPUFBIBMemory : public IBMemory {
    public:
      GPUFBIBMemory(Memory _me, GPU *_gpu, CUdeviceptr _base, size_t _size);

    public:
      GPU *gpu;
      CUdeviceptr base;
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
      //off_t src_gpu_off, dst_gpu_off;
      GPU* dst_gpu;
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
      GPUTransferCompletion(XferDes *_xd, int _read_port_idx,
                            size_t _read_offset, size_t _read_size,
                            int _write_port_idx, size_t _write_offset,
                            size_t _write_size);

      virtual void request_completed(void);

    protected:
      XferDes *xd;
      int read_port_idx;
      size_t read_offset, read_size;
      int write_port_idx;
      size_t write_offset, write_size;
    };

    class MemSpecificCudaArray : public MemSpecificInfo {
    public:
      MemSpecificCudaArray(CUarray _array);
      virtual ~MemSpecificCudaArray();

      CUarray array;
    };

    class AddressInfoCudaArray : public TransferIterator::AddressInfoCustom {
    public:
      virtual int set_rect(const RegionInstanceImpl *inst,
                           const InstanceLayoutPieceBase *piece,
                           size_t field_size, size_t field_offset,
                           int ndims,
                           const int64_t lo[/*ndims*/],
                           const int64_t hi[/*ndims*/],
                           const int order[/*ndims*/]);

      CUarray array;
      int dim;
      size_t pos[3];
      size_t width_in_bytes, height, depth;
    };

    class GPUChannel;

    class GPUXferDes : public XferDes {
    public:
      GPUXferDes(uintptr_t _dma_op, Channel *_channel,
		 NodeID _launch_node, XferDesID _guid,
		 const std::vector<XferDesPortInfo>& inputs_info,
		 const std::vector<XferDesPortInfo>& outputs_info,
		 int _priority);

      long get_requests(Request** requests, long nr);

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

      // multi-threading of cuda copies for a given device is disabled by
      //  default (can be re-enabled with -cuda:mtdma 1)
      static const bool is_ordered = true;

      virtual bool needs_wrapping_iterator() const;
      virtual Memory suggest_ib_memories(Memory memory) const;

      virtual RemoteChannelInfo *construct_remote_info() const;

      virtual uint64_t
      supports_path(ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
                    CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
                    size_t total_bytes, const std::vector<size_t> *src_frags,
                    const std::vector<size_t> *dst_frags, XferDesKind *kind_ret = 0,
                    unsigned *bw_ret = 0, unsigned *lat_ret = 0);

      virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                       XferDesID guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int priority, XferDesRedopInfo redop_info,
                                       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request **requests, long nr);

    protected:
      friend class GPUIndirectXferDes;
      GPU *src_gpu;
    };

    class GPUIndirectRemoteChannelInfo : public SimpleRemoteChannelInfo {
    public:
      GPUIndirectRemoteChannelInfo(NodeID _owner, XferDesKind _kind,
                                   uintptr_t _remote_ptr,
                                   const std::vector<Channel::SupportedPath> &_paths);

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
      GPUIndirectRemoteChannel(uintptr_t _remote_ptr);
      virtual Memory suggest_ib_memories(Memory memory) const;
      virtual uint64_t
      supports_path(ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
                    CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
                    size_t total_bytes, const std::vector<size_t> *src_frags,
                    const std::vector<size_t> *dst_frags, XferDesKind *kind_ret = 0,
                    unsigned *bw_ret = 0, unsigned *lat_ret = 0);
      virtual bool needs_wrapping_iterator() const;
    };

    class GPUChannel : public SingleXDQChannel<GPUChannel, GPUXferDes> {
    public:
      GPUChannel(GPU* _src_gpu, XferDesKind _kind,
		 BackgroundWorkManager *bgwork);
      ~GPUChannel();

      // multi-threading of cuda copies for a given device is disabled by
      //  default (can be re-enabled with -cuda:mtdma 1)
      static const bool is_ordered = true;

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request** requests, long nr);

    private:
      GPU* src_gpu;
      //std::deque<Request*> pending_copies;
    };

    class GPUfillChannel;

    class GPUfillXferDes : public XferDes {
    public:
      GPUfillXferDes(uintptr_t _dma_op, Channel *_channel,
		     NodeID _launch_node, XferDesID _guid,
		     const std::vector<XferDesPortInfo>& inputs_info,
		     const std::vector<XferDesPortInfo>& outputs_info,
		     int _priority,
		     const void *_fill_data, size_t _fill_size,
                     size_t _fill_total);

      long get_requests(Request** requests, long nr);

      bool progress_xd(GPUfillChannel *channel, TimeLimit work_until);

    protected:
      size_t reduced_fill_size;
    };

    class GPUfillChannel : public SingleXDQChannel<GPUfillChannel, GPUfillXferDes> {
    public:
      GPUfillChannel(GPU* _gpu, BackgroundWorkManager *bgwork);

      // multiple concurrent cuda fills ok
      static const bool is_ordered = false;

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request** requests, long nr);

    protected:
      friend class GPUfillXferDes;

      GPU* gpu;
    };

    class GPUreduceChannel;

    class GPUreduceXferDes : public XferDes {
    public:
      GPUreduceXferDes(uintptr_t _dma_op, Channel *_channel,
                       NodeID _launch_node, XferDesID _guid,
                       const std::vector<XferDesPortInfo>& inputs_info,
                       const std::vector<XferDesPortInfo>& outputs_info,
                       int _priority,
                       XferDesRedopInfo _redop_info);

      long get_requests(Request** requests, long nr);

      bool progress_xd(GPUreduceChannel *channel, TimeLimit work_until);

    protected:
      XferDesRedopInfo redop_info;
      const ReductionOpUntyped *redop;
      CUfunction kernel;
      const void *kernel_host_proxy;
      GPUStream *stream;
    };

    class GPUreduceChannel : public SingleXDQChannel<GPUreduceChannel, GPUreduceXferDes> {
    public:
      GPUreduceChannel(GPU* _gpu, BackgroundWorkManager *bgwork);

      // multiple concurrent cuda reduces ok
      static const bool is_ordered = false;

      // helper method here so that GPUreduceRemoteChannel can use it too
      static bool is_gpu_redop(ReductionOpID redop_id);

      // override this because we have to be picky about which reduction ops
      //  we support
      virtual uint64_t supports_path(ChannelCopyInfo channel_copy_info,
                                     CustomSerdezID src_serdez_id,
                                     CustomSerdezID dst_serdez_id,
                                     ReductionOpID redop_id,
                                     size_t total_bytes,
                                     const std::vector<size_t> *src_frags,
                                     const std::vector<size_t> *dst_frags,
                                     XferDesKind *kind_ret = 0,
                                     unsigned *bw_ret = 0,
                                     unsigned *lat_ret = 0);

      virtual RemoteChannelInfo *construct_remote_info() const;

      virtual XferDes *create_xfer_des(uintptr_t dma_op,
				       NodeID launch_node,
				       XferDesID guid,
				       const std::vector<XferDesPortInfo>& inputs_info,
				       const std::vector<XferDesPortInfo>& outputs_info,
				       int priority,
				       XferDesRedopInfo redop_info,
				       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request** requests, long nr);

    protected:
      friend class GPUreduceXferDes;

      GPU* gpu;
    };

    class GPUreduceRemoteChannelInfo : public SimpleRemoteChannelInfo {
    public:
      GPUreduceRemoteChannelInfo(NodeID _owner, XferDesKind _kind,
                                 uintptr_t _remote_ptr,
                                 const std::vector<Channel::SupportedPath>& _paths);

      virtual RemoteChannel *create_remote_channel();

      template <typename S>
      bool serialize(S& serializer) const;

      template <typename S>
      static RemoteChannelInfo *deserialize_new(S& deserializer);

    protected:
      static Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                      GPUreduceRemoteChannelInfo> serdez_subclass;
    };

    class GPUreduceRemoteChannel : public RemoteChannel {
      friend class GPUreduceRemoteChannelInfo;

      GPUreduceRemoteChannel(uintptr_t _remote_ptr);

      virtual uint64_t supports_path(ChannelCopyInfo channel_copy_info,
                                     CustomSerdezID src_serdez_id,
                                     CustomSerdezID dst_serdez_id,
                                     ReductionOpID redop_id,
                                     size_t total_bytes,
                                     const std::vector<size_t> *src_frags,
                                     const std::vector<size_t> *dst_frags,
                                     XferDesKind *kind_ret = 0,
                                     unsigned *bw_ret = 0,
                                     unsigned *lat_ret = 0);
    };

    // active message for establishing cuda ipc mappings
    struct CudaIpcImportRequest {
      unsigned count = 0;
#if !defined(REALM_IS_WINDOWS)
      long hostid = 0;
#endif
      static void handle_message(NodeID sender, const CudaIpcImportRequest &args,
                                 const void *data, size_t datalen);
    };

    class GPUReplHeapListener : public ReplicatedHeap::Listener {
    public:
      GPUReplHeapListener(CudaModule *_module);

      virtual void chunk_created(void *base, size_t bytes);
      virtual void chunk_destroyed(void *base, size_t bytes);

    protected:
      CudaModule *module;
    };

    /// @brief Class for managing the lifetime of a given gpu allocation.  As instances of
    /// this class own an underlying resource they are not copyable and must be
    /// std::move'd (thus invalidating the original variable) or references made
    class GPUAllocation {
    public:
      // -- Constructors --
      GPUAllocation(void) = default;
      GPUAllocation(GPUAllocation &&other) noexcept;
      GPUAllocation(const GPUAllocation &) = delete;
      GPUAllocation &operator=(GPUAllocation &&) noexcept;
      GPUAllocation &operator=(const GPUAllocation &) = delete;
      ~GPUAllocation();

      // --- Accessors ---
      inline operator bool(void) const { return dev_ptr != 0; }
      /// @brief Accessor for the file descriptor or win32 HANDLE associated with the
      /// allocation.  This handle can be shared with other APIs or other processes and
      /// opened with GPUAllocation::open_handle.
      /// @note it is the caller's responsibility to close the handle afterward to prevent
      /// resource leaks.
      OsHandle get_os_handle(void) const;

      /// @brief Retrieves the CUipcMemHandle for this allocation that can be used with
      /// GPUAllocation::open_ipc
      /// @param handle The CUipcMemHandle associated with this allocation
      /// @return True if this allocation has an IPC handle to retrieve, false otherwise.
      inline bool get_ipc_handle(CUipcMemHandle &handle) const
      {
        if(has_ipc_handle) {
          handle = ipc_handle;
        }
        return has_ipc_handle;
      }
#if CUDA_VERSION >= 12030
      /// @brief Retrieves the CUmemFabricHandle associated with the allocation that can
      /// be used with GPUAllocation::open_fabric
      /// @param handle The CUmemFabricHandle associated with this allocation
      /// @return True if the allocation has a fabric handle to retrieve, false otherwise.
      bool get_fabric_handle(CUmemFabricHandle &handle) const;
#endif
      /// @brief Retrieves the base CUdeviceptr for the associated allocation that can be
      /// used to access the underlying memory of the allocation from the device or with
      /// cuda apis that take a CUdeviceptr.
      /// @note This device pointer is relative to the owning GPU, not to other GPUs. This
      /// is typically not an issue unless CUDA's unified virtual addressing is not
      /// available, which for almost all supported systems is always the case and is
      /// detected and handled elsewhere.
      /// @return The device address to use.
      inline CUdeviceptr get_dptr(void) const { return dev_ptr; }
      /// @brief Retrieves the owning GPU.
      /// @return GPU that owns this allocation
      inline GPU *get_gpu(void) const { return gpu; }
      /// @brief Retrieves the given size of the allocation
      /// @return The size of this allocation
      inline size_t get_size(void) const { return size; }

      /// @brief Retrieves the CPU accessible base address for the allocation, or nullptr
      /// if there is no way to access this allocation from the CPU
      /// @tparam T The type of the return type to cast to
      /// @return The CPU visible address for accessing this allocation, or nullptr if no
      /// such access is possible
      template <typename T = void>
      T *get_hptr(void) const
      {
        return static_cast<T *>(host_ptr);
      }

      /// @brief Retrieves the default win32 shared attributes for creating a shared
      /// object that can be set in CUmemAllocationProp::win32Metadata and passed to
      /// GPUAllocation::allocate_mmap.
      /// @return A pointer to the default shared attributes.  This pointer is internally
      /// managed and should never be freed
      static void *get_win32_shared_attributes(void);

      // -- Allocators --

      /// @brief Allocates device-located memory for the given gpu with the given size and
      /// features
      /// @param gpu GPU this allocation is destined for and for which it's lifetime is
      /// assured
      /// @param size Size of the requested allocation
      /// @param peer_enabled True if the allocation needs to be accessible via all the
      /// GPU's peers
      /// @param shareable True if the allocation needs to be shareable
      /// @return The GPUAllocation, or nullptr if unsuccessful
      static GPUAllocation *allocate_dev(GPU *gpu, size_t size, bool peer_enabled = true,
                                         bool shareable = true);
#if CUDA_VERSION >= 11000
      /// @brief Allocates memory based on the CUmemAllocationProp passed to it, and maps
      /// the memory according to the given vaddr or a newly retrieve vaddr if one is not
      /// provided.
      /// @param gpu GPU this allocation is destined for and for which it's lifetime is
      /// assured
      /// @param prop Memory properties to be passed to cuMemCreate for allocation
      /// @param size Size of the requested allocation
      /// @param vaddr An externally managed virtual address to map the created allocation
      /// to.  Passing zero here indicates that a new virtual address range should be
      /// requested and assigned, which can be retrieved via GPUAllocation::get_dptr
      /// @param peer_enabled True if the allocation needs to be accessible via all the
      /// GPU's peers
      /// @return The GPUAllocation, or nullptr if unsuccessful
      static GPUAllocation *allocate_mmap(GPU *gpu, const CUmemAllocationProp &prop,
                                          size_t size, CUdeviceptr vaddr = 0,
                                          bool peer_enabled = true);
#endif
      /// @brief Allocate CPU-located memory for the given gpu with the given size and
      /// features
      /// @param gpu GPU this allocation is destined for and for which it's lifetime is
      /// assured
      /// @param size Size of the requested allocation
      /// @param peer_enabled True if the allocation needs to be accessible via all the
      /// GPU's peers
      /// @param shareable True if the allocation needs to be shareable
      /// @param same_va True if the allocation must have the same GPU and CPU virtual
      /// addresses (e.g. get_dptr() == get_hptr()).
      /// @return The GPUAllocation, or nullptr if unsuccessful
      static GPUAllocation *allocate_host(GPU *gpu, size_t size, bool peer_enabled = true,
                                          bool shareable = true, bool same_va = true);
      /// @brief Allocate migratable memory that can be used with CUDA's managed memory
      /// APIs (cuMemPrefetchAsync, etc).
      /// @param gpu GPU this allocation is destined for and for which it's lifetime is
      /// assured
      /// @param size Size of the requested allocation
      /// @return The GPUAllocation, or nullptr if unsuccessful
      static GPUAllocation *allocate_managed(GPU *gpu, size_t size);
      /// @brief Create an allocation that registers the given CPU address range with
      /// CUDA, making it accessible from the device.
      /// @note This object instance only manages the lifetime of the registration to
      /// CUDA.  The given address range must be managed externally.
      /// @param gpu GPU this allocation is destined for and for which it's lifetime is
      /// assured
      /// @param ptr Base address to register.
      /// @param size Size of the requested allocation
      /// @param peer_enabled True if this memory needs to be accessible by this GPU's
      /// peers
      /// @return The GPUAllocation, or nullptr if unsuccessful
      static GPUAllocation *register_allocation(GPU *gpu, void *ptr, size_t size,
                                                bool peer_enabled = true);
      /// @brief Retrieves the GPUAllocation given the CUipcMemHandle
      /// @param gpu GPU this allocation is destined for and for which it's lifetime is
      /// assured
      /// @param mem_hdl CUipcMemHandle e.g. retrieved from GPUAllocation::get_ipc_handle
      /// @return The GPUAllocation, or nullptr if unsuccessful
      static GPUAllocation *open_ipc(GPU *gpu, CUipcMemHandle mem_hdl);
      /// @brief Retrieves the GPUAllocation given the OsHandle
      /// @param gpu GPU this allocation is destined for and for which it's lifetime is
      /// assured
      /// @param hdl The OsHandle e.g. retrieved from GPUAllocation::get_os_handle
      /// @param size Size of the requested allocation
      /// @param peer_enabled True if this memory needs to be accessible by this GPU's
      /// peers
      /// @return The GPUAllocation, or nullptr if unsuccessful
      static GPUAllocation *open_handle(GPU *gpu, OsHandle hdl, size_t size,
                                        bool peer_enabled = true);
#if CUDA_VERSION >= 12030
      /// @brief Retrieves the GPUAllocaiton given the fabric handle
      /// @param gpu GPU this allocation is destined for and for which it's lifetime is
      /// assured
      /// @param hdl The OsHandle e.g. retrieved from GPUAllocation::get_os_handle
      /// @param size Size of the requested allocation
      /// @param peer_enabled True if this memory needs to be accessible by this GPU's
      /// peers
      /// @return The GPUAllocation, or nullptr if unsuccessful
      static GPUAllocation *open_fabric(GPU *gpu, CUmemFabricHandle &hdl, size_t size,
                                        bool peer_enabled = true);
#endif

    private:
      CUresult map_allocation(GPU *gpu, CUmemGenericAllocationHandle handle, size_t size,
                              CUdeviceptr va = 0, size_t offset = 0,
                              bool peer_enabled = false);

#if CUDA_VERSION >= 11000
      /// @brief Helper function to return the aligned size for the allocation given the
      /// prop.
      /// @param prop Properties of the memory for which to retrieve alignment for
      /// @param size Requested size of the allocation
      /// @return Aligned size of the request
      static size_t align_size(const CUmemAllocationProp &prop, size_t size);
#endif
      // -- Deleters --
      typedef void (*DeleterCallback)(GPUAllocation &alloc);

      // These are helper functions to manage what freeing strategy needs to be used to
      // properly free the allocation
      static void cuda_malloc_free(GPUAllocation &alloc);
      static void cuda_malloc_host_free(GPUAllocation &alloc);
      static void cuda_register_free(GPUAllocation &alloc);
      static void cuda_ipc_free(GPUAllocation &alloc);
#if CUDA_VERSION >= 11000
      static void cuda_memmap_free(GPUAllocation &alloc);
#endif

      // -- Members --
      /// Owning GPU.
      GPU *gpu = nullptr;
      /// Device-accessible address relative to the owning GPU
      CUdeviceptr dev_ptr = 0;
      /// CPU-accessible address
      void *host_ptr = nullptr;
      /// Size of the allocation
      size_t size = 0;
      /// The chosen deleter strategy for the underlying allocation mechanism
      DeleterCallback deleter = nullptr;
#if CUDA_VERSION >= 11000
      /// The generic allocation handle
      CUmemGenericAllocationHandle mmap_handle = 0;
      // True if VA needs to be released for cuMemMap'ed memory
      // or if the registered memory actually needs to be unregistered
      bool owns_va = true;
#endif
      /// True if GPUAllocation::ipc_handle is a valid handle
      bool has_ipc_handle = false;
      /// The ipc_handle that can be used with cuIpc* APIs.
      CUipcMemHandle ipc_handle;
    };

#ifdef REALM_CUDA_DYNAMIC_LOAD
  // cuda driver and/or runtime entry points
#define CUDA_DRIVER_FNPTR(name) (name##_fnptr)

#define CUDA_DRIVER_APIS(__op__)                                                         \
  __op__(cuModuleGetFunction);                                                           \
  __op__(cuCtxGetDevice);                                                                \
  __op__(cuCtxEnablePeerAccess);                                                         \
  __op__(cuCtxGetFlags);                                                                 \
  __op__(cuCtxGetStreamPriorityRange);                                                   \
  __op__(cuCtxPopCurrent);                                                               \
  __op__(cuCtxPushCurrent);                                                              \
  __op__(cuCtxSynchronize);                                                              \
  __op__(cuDeviceCanAccessPeer);                                                         \
  __op__(cuDeviceGet);                                                                   \
  __op__(cuDeviceGetUuid);                                                               \
  __op__(cuDeviceGetAttribute);                                                          \
  __op__(cuDeviceGetCount);                                                              \
  __op__(cuDeviceGetName);                                                               \
  __op__(cuDevicePrimaryCtxRelease);                                                     \
  __op__(cuDevicePrimaryCtxRetain);                                                      \
  __op__(cuDevicePrimaryCtxSetFlags);                                                    \
  __op__(cuDeviceTotalMem);                                                              \
  __op__(cuEventCreate);                                                                 \
  __op__(cuEventDestroy);                                                                \
  __op__(cuEventQuery);                                                                  \
  __op__(cuEventRecord);                                                                 \
  __op__(cuGetErrorName);                                                                \
  __op__(cuGetErrorString);                                                              \
  __op__(cuInit);                                                                        \
  __op__(cuIpcCloseMemHandle);                                                           \
  __op__(cuIpcGetMemHandle);                                                             \
  __op__(cuIpcOpenMemHandle);                                                            \
  __op__(cuLaunchKernel);                                                                \
  __op__(cuMemAllocManaged);                                                             \
  __op__(cuMemAlloc);                                                                    \
  __op__(cuMemcpy2DAsync);                                                               \
  __op__(cuMemcpy3DAsync);                                                               \
  __op__(cuMemcpyAsync);                                                                 \
  __op__(cuMemcpyDtoDAsync);                                                             \
  __op__(cuMemcpyDtoH);                                                                  \
  __op__(cuMemcpyDtoHAsync);                                                             \
  __op__(cuMemcpyHtoD);                                                                  \
  __op__(cuMemcpyHtoDAsync);                                                             \
  __op__(cuMemFreeHost);                                                                 \
  __op__(cuMemFree);                                                                     \
  __op__(cuMemGetInfo);                                                                  \
  __op__(cuMemHostAlloc);                                                                \
  __op__(cuMemHostGetDevicePointer);                                                     \
  __op__(cuMemHostRegister);                                                             \
  __op__(cuMemHostUnregister);                                                           \
  __op__(cuMemsetD16Async);                                                              \
  __op__(cuMemsetD2D16Async);                                                            \
  __op__(cuMemsetD2D32Async);                                                            \
  __op__(cuMemsetD2D8Async);                                                             \
  __op__(cuMemsetD32Async);                                                              \
  __op__(cuMemsetD8Async);                                                               \
  __op__(cuModuleLoadDataEx);                                                            \
  __op__(cuStreamAddCallback);                                                           \
  __op__(cuStreamCreate);                                                                \
  __op__(cuStreamCreateWithPriority);                                                    \
  __op__(cuStreamDestroy);                                                               \
  __op__(cuStreamSynchronize);                                                           \
  __op__(cuOccupancyMaxPotentialBlockSize);                                              \
  __op__(cuOccupancyMaxPotentialBlockSizeWithFlags);                                     \
  __op__(cuEventSynchronize);                                                            \
  __op__(cuEventElapsedTime);                                                            \
  __op__(cuOccupancyMaxActiveBlocksPerMultiprocessor);                                   \
  __op__(cuMemAddressReserve);                                                           \
  __op__(cuMemAddressFree);                                                              \
  __op__(cuMemCreate);                                                                   \
  __op__(cuMemRelease);                                                                  \
  __op__(cuMemMap);                                                                      \
  __op__(cuMemUnmap);                                                                    \
  __op__(cuMemSetAccess);                                                                \
  __op__(cuMemGetAllocationGranularity);                                                 \
  __op__(cuMemExportToShareableHandle);                                                  \
  __op__(cuMemImportFromShareableHandle);                                                \
  __op__(cuStreamWaitEvent);                                                             \
  __op__(cuStreamQuery);                                                                 \
  __op__(cuMemGetAddressRange);                                                          \
  __op__(cuPointerGetAttributes);                                                        \
  __op__(cuLaunchHostFunc)

#if CUDA_VERSION >= 11030
// cuda 11.3+ gives us handy PFN_... types
#define DECL_FNPTR_EXTERN(name) extern PFN_##name name##_fnptr;
#else
    // before cuda 11.3, we have to rely on typeof/decltype
#define DECL_FNPTR_EXTERN(name) extern decltype(&name) name##_fnptr;
#endif
    CUDA_DRIVER_APIS(DECL_FNPTR_EXTERN);
  #undef DECL_FNPTR_EXTERN

#else
  #define CUDA_DRIVER_FNPTR(name) (name)
#endif

#define NVML_FNPTR(name) (name##_fnptr)

#if NVML_API_VERSION >= 11
#define NVML_11_APIS(__op__) __op__(nvmlDeviceGetMemoryAffinity);
#else
#define NVML_11_APIS(__op__)
#endif

#if CUDA_VERSION < 11040
    // Define an NVML api that doesn't exist prior to CUDA Toolkit 11.5, but should
    // exist in systems that require it that we need to support (we'll detect it's
    // availability later)
    //
    // Although these are NVML apis, NVML_API_VERSION doesn't support any way to detect
    // minor versioning, so we'll use the cuda header's versioning here, which should
    // coincide with the versions we're looking for
    typedef enum nvmlIntNvLinkDeviceType_enum
    {
      NVML_NVLINK_DEVICE_TYPE_GPU = 0x00,
      NVML_NVLINK_DEVICE_TYPE_IBMNPU = 0x01,
      NVML_NVLINK_DEVICE_TYPE_SWITCH = 0x02,
      NVML_NVLINK_DEVICE_TYPE_UNKNOWN = 0xFF
    } nvmlIntNvLinkDeviceType_t;

    nvmlReturn_t
    nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link,
                                        nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType);
#endif

#define NVML_APIS(__op__)                                                                \
  __op__(nvmlInit);                                                                      \
  __op__(nvmlDeviceGetHandleByUUID);                                                     \
  __op__(nvmlDeviceGetMaxPcieLinkWidth);                                                 \
  __op__(nvmlDeviceGetMaxPcieLinkGeneration);                                            \
  __op__(nvmlDeviceGetNvLinkState);                                                      \
  __op__(nvmlDeviceGetNvLinkVersion);                                                    \
  __op__(nvmlDeviceGetNvLinkRemotePciInfo);                                              \
  __op__(nvmlDeviceGetNvLinkRemoteDeviceType);                                           \
  NVML_11_APIS(__op__);

#define DECL_FNPTR_EXTERN(name) extern decltype(&name) name##_fnptr;
    NVML_APIS(DECL_FNPTR_EXTERN)
#undef DECL_FNPTR_EXTERN

  }; // namespace Cuda

}; // namespace Realm

#endif
