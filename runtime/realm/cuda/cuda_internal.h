/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include "realm/realm_config.h"

#include <cuda.h>

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
#define REPORT_CU_ERROR(cmd, ret) \
  do { \
    const char *name, *str; \
    CUDA_DRIVER_FNPTR(cuGetErrorName)(ret, &name);      \
    CUDA_DRIVER_FNPTR(cuGetErrorString)(ret, &str);                 \
    fprintf(stderr, "CU: %s = %d (%s): %s\n", cmd, ret, name, str); \
    abort(); \
  } while(0)
#else
  do { \
    fprintf(stderr, "CU: %s = %d\n", cmd, ret); \
    abort(); \
  } while(0)
#endif

#define CHECK_CU(cmd) do {                      \
  CUresult ret = (cmd); \
  if(ret != CUDA_SUCCESS) REPORT_CU_ERROR(#cmd, ret); \
} while(0)

#ifdef REALM_USE_CUDART_HIJACK
#define CUDA_DEVICE_ATTRIBUTES_PRE_8_0(__op__) \
  __op__(major, COMPUTE_CAPABILITY_MAJOR)                \
  __op__(minor, COMPUTE_CAPABILITY_MINOR)                \
  __op__(sharedMemPerBlock, MAX_SHARED_MEMORY_PER_BLOCK) \
  __op__(regsPerBlock,      MAX_REGISTERS_PER_BLOCK)     \
  __op__(warpSize,          WARP_SIZE)                   \
  __op__(memPitch, MAX_PITCH)                            \
  __op__(maxThreadsPerBlock, MAX_THREADS_PER_BLOCK)      \
  __op__(maxThreadsDimX, MAX_BLOCK_DIM_X)                \
  __op__(maxThreadsDimY, MAX_BLOCK_DIM_Y)                \
  __op__(maxThreadsDimZ, MAX_BLOCK_DIM_Z)                \
  __op__(maxGridSizeX, MAX_GRID_DIM_X)                   \
  __op__(maxGridSizeY, MAX_GRID_DIM_Y)                   \
  __op__(maxGridSizeZ, MAX_GRID_DIM_Z)                   \
  __op__(clockRate, CLOCK_RATE)                          \
  __op__(totalConstMem, TOTAL_CONSTANT_MEMORY)           \
  __op__(deviceOverlap, GPU_OVERLAP)                     \
  __op__(multiProcessorCount, MULTIPROCESSOR_COUNT)      \
  __op__(kernelExecTimeoutEnabled, KERNEL_EXEC_TIMEOUT)  \
  __op__(integrated, INTEGRATED)                         \
  __op__(canMapHostMemory, CAN_MAP_HOST_MEMORY)          \
  __op__(computeMode, COMPUTE_MODE)                      \
  __op__(concurrentKernels, CONCURRENT_KERNELS)          \
  __op__(ECCEnabled, ECC_ENABLED)                        \
  __op__(pciBusID, PCI_BUS_ID)                           \
  __op__(pciDeviceID, PCI_DEVICE_ID)                     \
  __op__(pciDomainID, PCI_DOMAIN_ID)                     \
  __op__(tccDriver, TCC_DRIVER)                          \
  __op__(asyncEngineCount, ASYNC_ENGINE_COUNT)           \
  __op__(unifiedAddressing, UNIFIED_ADDRESSING)          \
  __op__(memoryClockRate, MEMORY_CLOCK_RATE)             \
  __op__(memoryBusWidth, GLOBAL_MEMORY_BUS_WIDTH)        \
  __op__(l2CacheSize, L2_CACHE_SIZE)                     \
  __op__(maxThreadsPerMultiProcessor, MAX_THREADS_PER_MULTIPROCESSOR) \
  __op__(streamPrioritiesSupported, STREAM_PRIORITIES_SUPPORTED)      \
  __op__(globalL1CacheSupported, GLOBAL_L1_CACHE_SUPPORTED)           \
  __op__(localL1CacheSupported, LOCAL_L1_CACHE_SUPPORTED)             \
  __op__(sharedMemPerMultiprocessor, MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)  \
  __op__(regsPerMultiprocessor, MAX_REGISTERS_PER_MULTIPROCESSOR)           \
  __op__(managedMemory, MANAGED_MEMORY)                                     \
  __op__(isMultiGpuBoard, MULTI_GPU_BOARD)                                  \
  __op__(multiGpuBoardGroupID, MULTI_GPU_BOARD_GROUP_ID)

#if CUDA_VERSION >= 8000
#define CUDA_DEVICE_ATTRIBUTES_8_0(__op__)                                        \
  __op__(singleToDoublePrecisionPerfRatio, SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO) \
  __op__(pageableMemoryAccess, PAGEABLE_MEMORY_ACCESS)                            \
  __op__(concurrentManagedAccess, CONCURRENT_MANAGED_ACCESS)
#else
#define CUDA_DEVICE_ATTRIBUTES_8_0(__op__)
#endif

#if CUDA_VERSION >= 9000
#define CUDA_DEVICE_ATTRIBUTES_9_0(__op__)                                            \
  __op__(computePreemptionSupported, COMPUTE_PREEMPTION_SUPPORTED)                    \
  __op__(canUseHostPointerForRegisteredMem, CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM)  \
  __op__(cooperativeLaunch, COOPERATIVE_LAUNCH)                                       \
  __op__(cooperativeMultiDeviceLaunch, COOPERATIVE_MULTI_DEVICE_LAUNCH)               \
  __op__(sharedMemPerBlockOptin, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
#else
#define CUDA_DEVICE_ATTRIBUTES_9_0(__op__)
#endif

#if CUDA_VERSION >= 9200
#define CUDA_DEVICE_ATTRIBUTES_9_2(__op__)                                                      \
  __op__(pageableMemoryAccessUsesHostPageTables, PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES)  \
  __op__(directManagedMemAccessFromHost, DIRECT_MANAGED_MEM_ACCESS_FROM_HOST)
#else
#define CUDA_DEVICE_ATTRIBUTES_9_2(__op__)
#endif
#if CUDA_VERSION >= 11000
#define CUDA_DEVICE_ATTRIBUTES_11_0(__op__)                         \
  __op__(maxBlocksPerMultiProcessor, MAX_BLOCKS_PER_MULTIPROCESSOR) \
  __op__(accessPolicyMaxWindowSize, MAX_ACCESS_POLICY_WINDOW_SIZE)
#else
#define CUDA_DEVICE_ATTRIBUTES_11_0(__op__)
#endif

#define CUDA_DEVICE_ATTRIBUTES(__op__)   \
  CUDA_DEVICE_ATTRIBUTES_PRE_8_0(__op__) \
  CUDA_DEVICE_ATTRIBUTES_8_0 (__op__)    \
  CUDA_DEVICE_ATTRIBUTES_9_0 (__op__)    \
  CUDA_DEVICE_ATTRIBUTES_9_2 (__op__)    \
  CUDA_DEVICE_ATTRIBUTES_11_0(__op__)
#else
#define CUDA_DEVICE_ATTRIBUTES(__op__)      \
  __op__(major, COMPUTE_CAPABILITY_MAJOR)   \
  __op__(minor, COMPUTE_CAPABILITY_MINOR)
#endif

namespace Realm {

  namespace Cuda {

    struct GPUInfo 
    {
      int index;  // index used by CUDA runtime
      CUdevice device;
      #define DEFINE_CUDA_ATTRIBUTES(name, attr) int name;
      CUDA_DEVICE_ATTRIBUTES(DEFINE_CUDA_ATTRIBUTES)

#ifndef REALM_USE_CUDART_HIAJCK
      static const size_t MAX_NAME_LEN = 64;
      char name[MAX_NAME_LEN];

      size_t totalGlobalMem;
#endif
      std::set<CUdevice> peers;  // other GPUs we can do p2p copies with
    };

    enum GPUMemcpyKind {
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
    class GPUZCMemory;
    class GPUFBIBMemory;
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
      virtual ~GPUMemcpy(void) { }
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

      virtual void print(std::ostream& os) const;

      IntrusiveListLink<GPUWorkFence> fence_list_link;
      REALM_PMTA_DEFN(GPUWorkFence,IntrusiveListLink<GPUWorkFence>,fence_list_link);
      typedef IntrusiveList<GPUWorkFence, REALM_PMTA_USE(GPUWorkFence,fence_list_link), DummyLock> FenceList;

    protected:
      static void cuda_callback(CUstream stream, CUresult res, void *data);
    };

    class GPUWorkStart : public Realm::Operation::AsyncWorkItem {
    public:
      GPUWorkStart(Realm::Operation *op);

      virtual void request_cancellation(void) { return; };

      void enqueue_on_stream(GPUStream *stream);

      virtual void print(std::ostream& os) const;

      void mark_gpu_work_start();

    protected:
      static void cuda_start_callback(CUstream stream, CUresult res, void *data);
    };

    class GPUMemcpyFence : public GPUMemcpy {
    public:
      GPUMemcpyFence(GPU *_gpu, GPUMemcpyKind _kind,
		     GPUWorkFence *_fence);

      virtual void execute(GPUStream *stream);

    protected:
      GPUWorkFence *fence;
    };

    class GPUMemcpy1D : public GPUMemcpy {
    public:
      GPUMemcpy1D(GPU *_gpu,
		  void *_dst, const void *_src, size_t _bytes, GPUMemcpyKind _kind,
		  GPUCompletionNotification *_notification);

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
      GPUStream *local_stream;  // used by do_span
    };

    class GPUMemcpy2D : public GPUMemcpy {
    public:
      GPUMemcpy2D(GPU *_gpu,
                  void *_dst, const void *_src,
                  off_t _dst_stride, off_t _src_stride,
                  size_t _bytes, size_t _lines,
                  GPUMemcpyKind _kind,
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
      GPUMemcpy3D(GPU *_gpu,
                  void *_dst, const void *_src,
                  off_t _dst_stride, off_t _src_stride,
                  off_t _dst_pstride, off_t _src_pstride,
                  size_t _bytes, size_t _height, size_t _depth,
                  GPUMemcpyKind _kind,
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
      GPUMemset1D(GPU *_gpu,
		  void *_dst, size_t _bytes,
		  const void *_fill_data, size_t _fill_data_size,
		  GPUCompletionNotification *_notification);

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
      GPUMemset2D(GPU *_gpu,
		  void *_dst, size_t _dst_stride,
		  size_t _bytes, size_t _lines,
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
      GPUMemset3D(GPU *_gpu,
		  void *_dst, size_t _dst_stride, size_t _dst_pstride,
		  size_t _bytes, size_t _height, size_t _depth,
		  const void *_fill_data, size_t _fill_data_size,
		  GPUCompletionNotification *_notification);

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
      void wait_on_streams(const std::set<GPUStream*> &other_streams);

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
		     GPUCompletionNotification *notification=NULL, GPUWorkStart *start=NULL);

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

      bool can_access_peer(GPU *peer);

      GPUStream *find_stream(CUstream stream) const;
      GPUStream *get_null_task_stream(void) const;
      GPUStream *get_next_task_stream(bool create = false);
      GPUStream *get_next_d2d_stream();
    protected:
      CUmodule load_cuda_module(const void *data);

    public:
      CudaModule *module;
      GPUInfo *info;
      GPUWorker *worker;
      GPUProcessor *proc;
      GPUFBMemory *fbmem;
      GPUFBIBMemory *fb_ibmem;

      CUcontext context;
      CUdeviceptr fbmem_base, fb_ibmem_base;

      // which system memories have been registered and can be used for cuMemcpyAsync
      std::set<Memory> pinned_sysmems;

      // managed memories we can concurrently access
      std::set<Memory> managed_mems;

      // which other FBs we have peer access to
      std::set<Memory> peer_fbs;

      // streams for different copy types and a pile for actual tasks
      GPUStream *host_to_device_stream;
      GPUStream *device_to_host_stream;
      GPUStream *device_to_device_stream;
      std::vector<GPUStream *> device_to_device_streams;
      std::vector<GPUStream *> peer_to_peer_streams; // indexed by target
      std::vector<GPUStream *> task_streams;
      atomic<unsigned> next_task_stream, next_d2d_stream;

      GPUEventPool event_pool;

      // this can technically be different in each context (but probably isn't
      //  in practice)
      int least_stream_priority, greatest_stream_priority;

      struct CudaIpcMapping {
        NodeID owner;
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
      virtual void shutdown(void);

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
      void event_create(CUevent *event, int flags);
      void event_destroy(CUevent event);
      void event_record(CUevent event, CUstream stream);
      void event_synchronize(CUevent event);
      void event_elapsed_time(float *ms, CUevent start, CUevent end);
      
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
      virtual uint64_t supports_path(Memory src_mem, Memory dst_mem,
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

      virtual uint64_t supports_path(Memory src_mem, Memory dst_mem,
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


    // active messages for establishing cuda ipc mappings

    struct CudaIpcRequest {
#ifdef REALM_ON_LINUX
      long hostid;  // POSIX hostid
#endif

      static void handle_message(NodeID sender, const CudaIpcRequest& args,
                                 const void *data, size_t datalen);
    };

    struct CudaIpcResponse {
      unsigned count;

      static void handle_message(NodeID sender, const CudaIpcResponse& args,
                                 const void *data, size_t datalen);
    };

    struct CudaIpcRelease {

      static void handle_message(NodeID sender, const CudaIpcRelease& args,
                                 const void *data, size_t datalen);
    };


#ifdef REALM_CUDA_DYNAMIC_LOAD
  // cuda driver and/or runtime entry points
#define CUDA_DRIVER_FNPTR(name) (name##_fnptr)

#define CUDA_DRIVER_APIS(__op__)                                               \
  __op__(cuModuleGetFunction);                                                 \
  __op__(cuCtxEnablePeerAccess);                                               \
  __op__(cuCtxGetFlags);                                                       \
  __op__(cuCtxGetStreamPriorityRange);                                         \
  __op__(cuCtxPopCurrent);                                                     \
  __op__(cuCtxPushCurrent);                                                    \
  __op__(cuCtxSynchronize);                                                    \
  __op__(cuDeviceCanAccessPeer);                                               \
  __op__(cuDeviceGet);                                                         \
  __op__(cuDeviceGetAttribute);                                                \
  __op__(cuDeviceGetCount);                                                    \
  __op__(cuDeviceGetName);                                                     \
  __op__(cuDevicePrimaryCtxRelease);                                           \
  __op__(cuDevicePrimaryCtxRetain);                                            \
  __op__(cuDevicePrimaryCtxSetFlags);                                          \
  __op__(cuDeviceTotalMem);                                                    \
  __op__(cuEventCreate);                                                       \
  __op__(cuEventDestroy);                                                      \
  __op__(cuEventQuery);                                                        \
  __op__(cuEventRecord);                                                       \
  __op__(cuGetErrorName);                                                      \
  __op__(cuGetErrorString);                                                    \
  __op__(cuInit);                                                              \
  __op__(cuIpcCloseMemHandle);                                                 \
  __op__(cuIpcGetMemHandle);                                                   \
  __op__(cuIpcOpenMemHandle);                                                  \
  __op__(cuLaunchKernel);                                                      \
  __op__(cuMemAllocManaged);                                                   \
  __op__(cuMemAlloc);                                                          \
  __op__(cuMemcpy2DAsync);                                                     \
  __op__(cuMemcpy3DAsync);                                                     \
  __op__(cuMemcpyAsync);                                                       \
  __op__(cuMemcpyDtoDAsync);                                                   \
  __op__(cuMemcpyDtoH);                                                        \
  __op__(cuMemcpyDtoHAsync);                                                   \
  __op__(cuMemcpyHtoD);                                                        \
  __op__(cuMemcpyHtoDAsync);                                                   \
  __op__(cuMemFreeHost);                                                       \
  __op__(cuMemFree);                                                           \
  __op__(cuMemGetInfo);                                                        \
  __op__(cuMemHostAlloc);                                                      \
  __op__(cuMemHostGetDevicePointer);                                           \
  __op__(cuMemHostRegister);                                                   \
  __op__(cuMemHostUnregister);                                                 \
  __op__(cuMemsetD16Async);                                                    \
  __op__(cuMemsetD2D16Async);                                                  \
  __op__(cuMemsetD2D32Async);                                                  \
  __op__(cuMemsetD2D8Async);                                                   \
  __op__(cuMemsetD32Async);                                                    \
  __op__(cuMemsetD8Async);                                                     \
  __op__(cuModuleLoadDataEx);                                                  \
  __op__(cuStreamAddCallback);                                                 \
  __op__(cuStreamCreate);                                                      \
  __op__(cuStreamCreateWithPriority);                                          \
  __op__(cuStreamDestroy);                                                     \
  __op__(cuStreamSynchronize);                                                 \
  __op__(cuStreamWaitEvent)

  #if CUDA_VERSION >= 11030
    // cuda 11.3+ gives us handy PFN_... types
    #define DECL_FNPTR_EXTERN(name) \
      extern PFN_ ## name name ## _fnptr;
  #else
    // before cuda 11.3, we have to rely on typeof/decltype
    #define DECL_FNPTR_EXTERN(name) \
      extern decltype(&name) name ## _fnptr;
  #endif
    CUDA_DRIVER_APIS(DECL_FNPTR_EXTERN);
  #undef DECL_FNPTR_EXTERN

#else
  #define CUDA_DRIVER_FNPTR(name) (name)
#endif

  }; // namespace Cuda

}; // namespace Realm 

#endif
