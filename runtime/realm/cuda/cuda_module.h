/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#ifndef LOWLEVEL_GPU_H
#define LOWLEVEL_GPU_H

#include <cuda.h>

// We don't actually use the cuda runtime, but
// we need all its declarations so we have all the right types
#include <cuda_runtime.h>

#include "realm/operation.h"
#include "realm/module.h"
#include "realm/threads.h"
#include "realm/circ_queue.h"
#include "realm/indexspace.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"

#define CHECK_CUDART(cmd) do { \
  cudaError_t ret = (cmd); \
  if(ret != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s)\n", #cmd, ret, cudaGetErrorString(ret)); \
    assert(0); \
    exit(1); \
  } \
} while(0)

// Need CUDA 6.5 or later for good error reporting
#if CUDA_VERSION >= 6050
#define CHECK_CU(cmd) do { \
  CUresult ret = (cmd); \
  if(ret != CUDA_SUCCESS) { \
    const char *name, *str; \
    cuGetErrorName(ret, &name); \
    cuGetErrorString(ret, &str); \
    fprintf(stderr, "CU: %s = %d (%s): %s\n", #cmd, ret, name, str); \
    assert(0); \
    exit(1); \
  } \
} while(0)
#else
#define CHECK_CU(cmd) do { \
  CUresult ret = (cmd); \
  if(ret != CUDA_SUCCESS) { \
    fprintf(stderr, "CU: %s = %d\n", #cmd, ret); \
    assert(0); \
    exit(1); \
  } \
} while(0)
#endif

namespace Realm {
  namespace Cuda {

    class GPU;
    class GPUWorker;
    struct GPUInfo;
    class GPUZCMemory;

    // our interface to the rest of the runtime
    class CudaModule : public Module {
    protected:
      CudaModule(void);
      
    public:
      virtual ~CudaModule(void);

      static Module *create_module(RuntimeImpl *runtime, std::vector<std::string>& cmdline);

      // do any general initialization - this is called after all configuration is
      //  complete
      virtual void initialize(RuntimeImpl *runtime);

      // create any memories provided by this module (default == do nothing)
      //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
      virtual void create_memories(RuntimeImpl *runtime);

      // create any processors provided by the module (default == do nothing)
      //  (each new ProcessorImpl should use a Processor from
      //   RuntimeImpl::next_local_processor_id)
      virtual void create_processors(RuntimeImpl *runtime);

      // create any DMA channels provided by the module (default == do nothing)
      virtual void create_dma_channels(RuntimeImpl *runtime);

      // create any code translators provided by the module (default == do nothing)
      virtual void create_code_translators(RuntimeImpl *runtime);

      // clean up any common resources created by the module - this will be called
      //  after all memories/processors/etc. have been shut down and destroyed
      virtual void cleanup(void);

    public:
      size_t cfg_zc_mem_size_in_mb, cfg_zc_ib_size_in_mb;
      size_t cfg_fb_mem_size_in_mb;
      unsigned cfg_num_gpus, cfg_gpu_streams;
      bool cfg_use_background_workers, cfg_use_shared_worker, cfg_pin_sysmem;
      bool cfg_fences_use_callbacks;
      bool cfg_suppress_hijack_warning;

      // "global" variables live here too
      GPUWorker *shared_worker;
      std::map<GPU *, GPUWorker *> dedicated_workers;
      std::vector<GPUInfo *> gpu_info;
      std::vector<GPU *> gpus;
      void *zcmem_cpu_base, *zcib_cpu_base;
      GPUZCMemory *zcmem;
      std::vector<void *> registered_host_ptrs;
    };

    REGISTER_REALM_MODULE(CudaModule);

    struct GPUInfo {
      int index;  // index used by CUDA runtime
      CUdevice device;

      static const size_t MAX_NAME_LEN = 64;
      char name[MAX_NAME_LEN];

      int compute_major, compute_minor;
      size_t total_mem;
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

    protected:
      static void cuda_callback(CUstream stream, CUresult res, void *data);
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
                  off_t _dst_height, off_t _src_height,
                  size_t _bytes, size_t _height, size_t _depth,
                  GPUMemcpyKind _kind,
                  GPUCompletionNotification *_notification);

      virtual ~GPUMemcpy3D(void);

    public:
      virtual void execute(GPUStream *stream);
    protected:
      void *dst;
      const void *src;
      off_t dst_stride, src_stride, dst_height, src_height;
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
	void *indirect;
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
	void *indirect;
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
      GPUStream(GPU *_gpu, GPUWorker *_worker);
      ~GPUStream(void);

      GPU *get_gpu(void) const;
      CUstream get_stream(void) const;

      // may be called by anybody to enqueue a copy or an event
      void add_copy(GPUMemcpy *copy);
      void add_fence(GPUWorkFence *fence);
      void add_notification(GPUCompletionNotification *notification);

      // to be called by a worker (that should already have the GPU context
      //   current) - returns true if any work remains
      bool issue_copies(void);
      bool reap_events(void);

    protected:
      void add_event(CUevent event, GPUWorkFence *fence, 
		     GPUCompletionNotification *notification);

      GPU *gpu;
      GPUWorker *worker;

      CUstream stream;

      GASNetHSL mutex;

#define USE_CQ
#ifdef USE_CQ
      Realm::CircularQueue<GPUMemcpy *> pending_copies;
#else
      std::deque<GPUMemcpy *> pending_copies;
#endif

      struct PendingEvent {
	CUevent event;
	GPUWorkFence *fence;
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
    class GPUWorker {
    public:
      GPUWorker(void);
      virtual ~GPUWorker(void);

      // adds a stream that has work to be done
      void add_stream(GPUStream *s);

      // processes work on streams, optionally sleeping for work to show up
      // returns true if work remains to be done
      bool process_streams(bool sleep_on_empty);

      void start_background_thread(Realm::CoreReservationSet& crs,
				   size_t stack_size);
      void shutdown_background_thread(void);

    public:
      void thread_main(void);

    protected:
      GASNetHSL lock;
      GASNetCondVar condvar;
      std::set<GPUStream *> active_streams;

      // used by the background thread (if any)
      Realm::CoreReservation *core_rsrv;
      Realm::Thread *worker_thread;
      bool worker_shutdown_requested;
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

      CUevent get_event(void);
      void return_event(CUevent e);

    protected:
      GASNetHSL mutex;
      int batch_size, current_size, total_size;
      std::vector<CUevent> available_events;
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
	  int num_streams);
      ~GPU(void);

      void push_context(void);
      void pop_context(void);

      void register_fat_binary(const FatBin *data);
      void register_variable(const RegisteredVariable *var);
      void register_function(const RegisteredFunction *func);

      CUfunction lookup_function(const void *func);
      CUdeviceptr lookup_variable(const void *var);

      void create_processor(RuntimeImpl *runtime, size_t stack_size);
      void create_fb_memory(RuntimeImpl *runtime, size_t size);

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

      void fence_to_fb(Realm::Operation *op);
      void fence_from_fb(Realm::Operation *op);
      void fence_within_fb(Realm::Operation *op);
      void fence_to_peer(Realm::Operation *op, GPU *dst);

      bool can_access_peer(GPU *peer);

      GPUStream *switch_to_next_task_stream(void);
      GPUStream *get_current_task_stream(void);

    protected:
      CUmodule load_cuda_module(const void *data);

    public:
      CudaModule *module;
      GPUInfo *info;
      GPUWorker *worker;
      GPUProcessor *proc;
      GPUFBMemory *fbmem;

      CUcontext context;
      CUdeviceptr fbmem_base;

      // which system memories have been registered and can be used for cuMemcpyAsync
      std::set<Memory> pinned_sysmems;

      // which other FBs we have peer access to
      std::set<Memory> peer_fbs;

      // streams for different copy types and a pile for actual tasks
      GPUStream *host_to_device_stream;
      GPUStream *device_to_host_stream;
      GPUStream *device_to_device_stream;
      GPUStream *peer_to_peer_stream;
      std::vector<GPUStream *> task_streams;
      unsigned current_stream;

      GPUEventPool event_pool;

      std::map<const FatBin *, CUmodule> device_modules;
      std::map<const void *, CUfunction> device_functions;
      std::map<const void *, CUdeviceptr> device_variables;
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

      // calls that come from the CUDA runtime API
      void push_call_configuration(dim3 grid_dim, dim3 block_dim,
                                   size_t shared_size, void *stream);
      void pop_call_configuration(dim3 *grid_dim, dim3 *block_dim,
                                  size_t *shared_size, void *stream);

      void stream_synchronize(cudaStream_t stream);
      void device_synchronize(void);

      void event_create(cudaEvent_t *event, int flags);
      void event_destroy(cudaEvent_t event);
      void event_record(cudaEvent_t event, cudaStream_t stream);
      void event_synchronize(cudaEvent_t event);
      void event_elapsed_time(float *ms, cudaEvent_t start, cudaEvent_t end);
      
      void configure_call(dim3 grid_dim, dim3 block_dim,
			  size_t shared_memory, cudaStream_t stream);
      void setup_argument(const void *arg, size_t size, size_t offset);
      void launch(const void *func);
      void launch_kernel(const void *func, dim3 grid_dim, dim3 block_dim, 
                         void **args, size_t shared_memory, cudaStream_t stream);

      void gpu_memcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind);
      void gpu_memcpy_async(void *dst, const void *src, size_t size,
			    cudaMemcpyKind kind, cudaStream_t stream);
      void gpu_memcpy_to_symbol(const void *dst, const void *src, size_t size,
				size_t offset, cudaMemcpyKind kind);
      void gpu_memcpy_to_symbol_async(const void *dst, const void *src, size_t size,
				      size_t offset, cudaMemcpyKind kind,
				      cudaStream_t stream);
      void gpu_memcpy_from_symbol(void *dst, const void *src, size_t size,
				  size_t offset, cudaMemcpyKind kind);
      void gpu_memcpy_from_symbol_async(void *dst, const void *src, size_t size,
					size_t offset, cudaMemcpyKind kind,
					cudaStream_t stream);

      void gpu_memset(void *dst, int value, size_t count);
      void gpu_memset_async(void *dst, int value, size_t count, cudaStream_t stream);
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
        cudaStream_t stream; 
        CallConfig(dim3 _grid, dim3 _block, size_t _shared, cudaStream_t _stream);
      };
      std::vector<LaunchConfig> launch_configs;
      std::vector<char> kernel_args;
      std::vector<CallConfig> call_configs;
    protected:
      Realm::CoreReservation *core_rsrv;
    };

    class GPUFBMemory : public MemoryImpl {
    public:
      GPUFBMemory(Memory _me, GPU *_gpu, CUdeviceptr _base, size_t _size);

      virtual ~GPUFBMemory(void);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      // these work, but they are SLOW
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      virtual int get_home_node(off_t offset, size_t size);

    public:
      GPU *gpu;
      CUdeviceptr base;
    };

    class GPUZCMemory : public MemoryImpl {
    public:
      GPUZCMemory(Memory _me, CUdeviceptr _gpu_base, void *_cpu_base, size_t _size);

      virtual ~GPUZCMemory(void);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      virtual int get_home_node(off_t offset, size_t size);

    public:
      CUdeviceptr gpu_base;
      char *cpu_base;
    };

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
