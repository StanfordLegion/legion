/* Copyright 2015 Stanford University, NVIDIA Corporation
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
      size_t cfg_zc_mem_size_in_mb, cfg_fb_mem_size_in_mb;
      unsigned cfg_num_gpus, cfg_gpu_streams;
      bool cfg_use_background_workers, cfg_use_shared_worker, cfg_pin_sysmem;
      bool cfg_fences_use_callbacks;

    protected:
      // "global" variables live here too
      GPUWorker *shared_worker;
      std::map<GPU *, GPUWorker *> dedicated_workers;
      std::vector<GPUInfo *> gpu_info;
      std::vector<GPU *> gpus;
    };

    struct GPUInfo {
      int index;  // index used by CUDA runtime
      CUdevice device;

      static const size_t MAX_NAME_LEN = 64;
      char name[MAX_NAME_LEN];

      int compute_major, compute_minor;
      size_t total_mem;
      CUdevprop props;
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

      GPUMemcpy1D(GPU *_gpu,
		  void *_dst, const void *_src, 
		  const ElementMask *_mask, size_t _elmt_size,
		  GPUMemcpyKind _kind,
		  GPUCompletionNotification *_notification);

      virtual ~GPUMemcpy1D(void);

    public:
      void do_span(off_t pos, size_t len);
      virtual void execute(GPUStream *stream);
    protected:
      void *dst;
      const void *src;
      const ElementMask *mask;
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

      void register_fat_binary(FatBin *data);
      void register_variable(RegisteredVariable *var);
      void register_function(RegisteredFunction *func);

      CUfunction lookup_function(const void *func);
      CUdeviceptr lookup_variable(const void *var);

      void create_dma_channels(Realm::RuntimeImpl *r);

      // copy operations are asynchronous - use a fence (of the right type)
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

      void copy_from_fb_2d(void *dst, off_t src_offset,
                           off_t dst_stride, off_t src_stride,
                           size_t bytes, size_t lines,
			   GPUCompletionNotification *notification = 0);

      void copy_within_fb_2d(off_t dst_offset, off_t src_offset,
                             off_t dst_stride, off_t src_stride,
                             size_t bytes, size_t lines,
			     GPUCompletionNotification *notification = 0);

      void copy_to_peer(GPU *dst, off_t dst_offset, 
                        off_t src_offset, size_t bytes,
			GPUCompletionNotification *notification = 0);

      void copy_to_peer_2d(GPU *dst, off_t dst_offset, off_t src_offset,
                           off_t dst_stride, off_t src_stride,
                           size_t bytes, size_t lines,
			   GPUCompletionNotification *notification = 0);

      void copy_to_fb(off_t dst_offset, const void *src,
		      const ElementMask *mask, size_t elmt_size,
		      GPUCompletionNotification *notification = 0);

      void copy_from_fb(void *dst, off_t src_offset,
			const ElementMask *mask, size_t elmt_size,
			GPUCompletionNotification *notification = 0);

      void copy_within_fb(off_t dst_offset, off_t src_offset,
			  const ElementMask *mask, size_t elmt_size,
			  GPUCompletionNotification *notification = 0);

      void fence_to_fb(Realm::Operation *op);
      void fence_from_fb(Realm::Operation *op);
      void fence_within_fb(Realm::Operation *op);
      void fence_to_peer(Realm::Operation *op, GPU *dst);

      bool can_access_peer(GPU *peer);

      GPUStream *switch_to_next_task_stream(void);
      GPUStream *get_current_task_stream(void);

    public:
      CudaModule *module;
      GPUInfo *info;
      GPUWorker *worker;
      GPUProcessor *proc;
      GPUFBMemory *fbmem;

      CUcontext context;

      // streams for different copy types and a pile for actual tasks
      GPUStream *host_to_device_stream;
      GPUStream *device_to_host_stream;
      GPUStream *device_to_device_stream;
      GPUStream *peer_to_peer_stream;
      std::vector<GPUStream *> task_streams;
      unsigned current_stream;

      GPUEventPool event_pool;

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

#if 0
    protected:
      void initialize_cuda_stuff(void);
      void cleanup_cuda_stuff(void);
#endif

    public:
      virtual void shutdown(void);

#if 0
      void *get_zcmem_cpu_base(void) const;
      void *get_fbmem_gpu_base(void) const;
      size_t get_zcmem_size(void) const;
      size_t get_fbmem_size(void) const;
    public:
    public:
      void register_host_memory(Realm::MemoryImpl *m);
      void enable_peer_access(GPUProcessor *peer);
      void handle_peer_access(CUcontext peer_ctx);
      void handle_complete_copy(GPUMemcpy *copy);
    public:
      void load_context(void);
#endif

      static GPUProcessor *get_current_gpu_proc(void);

      // calls that come from the CUDA runtime API
      void stream_synchronize(cudaStream_t stream);
      void device_synchronize(void);
      
      void configure_call(dim3 grid_dim, dim3 block_dim,
			  size_t shared_memory, cudaStream_t stream);
      void setup_argument(const void *arg, size_t size, size_t offset);
      void launch(const void *func);

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

    public:
      GPU *gpu;

      // data needed for kernel launches
      struct LaunchConfig {
        dim3 grid;
        dim3 block;
        size_t shared;
	LaunchConfig(dim3 _grid, dim3 _block, size_t _shared);
      };
      std::vector<LaunchConfig> launch_configs;
      std::vector<char> kernel_args;

    protected:
#if 0
      const int gpu_index;
      const size_t zcmem_size, fbmem_size;
      const size_t zcmem_reserve, fbmem_reserve;
      GPUWorker *gpu_worker;
      void *zcmem_cpu_base;
      void *zcmem_gpu_base;
      void *fbmem_gpu_base;
#endif
      Realm::CoreReservation *core_rsrv;
    protected:
#if 0
      std::set<GPUProcessor*> peer_gpus;
#endif
    public:
#if 0
      // Our CUDA context that we will create
      CUdevice  proc_dev;
      CUcontext proc_ctx;

      GPUStream *switch_to_next_task_stream(void);
#endif
#if 0
    protected:
      size_t current_stream;
      std::vector<GPUStream *> task_streams;
#endif
    public:
      // Our helper cuda calls
      void** internal_register_fat_binary(void *fat_bin);
      void** internal_register_cuda_binary(void *cubin);
      void internal_unregister_fat_binary(void **fat_bin);
      void internal_register_var(void **fat_bin, char *host_var, 
                                 const char *device_name, bool ext, 
                                 int size, bool constant, bool global);
      void internal_register_function(void **fat_bin ,const char *host_fun,
                                      const char *device_fun);
      char internal_init_module(void **fat_bin);
      void load_module(CUmodule *module, const void *image);
      void find_function_handle(const void *func, CUfunction *handle);
    public:
#if 0
      // Our cuda calls
      cudaError_t internal_configure_call(dim3 gird_dim, dim3 block_dim, size_t shared_mem);
      cudaError_t internal_setup_argument(const void *arg, size_t size, size_t offset);
      cudaError_t internal_launch(const void *func);
      cudaError_t internal_gpu_memcpy(void *dst, const void *src, size_t size, bool sync);
      cudaError_t internal_gpu_memcpy_to_symbol(const void *dst, const void *src, size_t size,
                                       size_t offset, cudaMemcpyKind kind, bool sync);
      cudaError_t internal_gpu_memcpy_from_symbol(void *dst, const void *src, size_t size,
                                         size_t offset, cudaMemcpyKind kind, bool sync);
#endif
    private:
      struct VarInfo {
      public:
        const char *name;
        CUdeviceptr ptr;
        size_t size;
      };
      struct ModuleInfo {
        CUmodule module;
        std::set<const void*> host_aliases;
        std::set<const void*> var_aliases;
      };
    private:
      // Support for our internal cuda runtime
      std::map<void** /*fatbin*/,ModuleInfo> modules;
      std::map<const void*,CUfunction> device_functions;
      std::map<const void*,VarInfo> device_variables;
      // Modules allocated just during this task's lifetime
      std::set<void**> task_modules;
    public:
      // Support for deferring loading of modules and functions
      // until after we have initialized the runtime
#if 0
      static std::map<void*,void**>& get_deferred_modules(void);
      static std::map<void*,void**>& get_deferred_cubins(void);
      static std::deque<DeferredFunction>& get_deferred_functions(void);
      static std::deque<DeferredVariable>& get_deferred_variables(void);
      static void** defer_module_load(void *fat_bin);
      static void** defer_cubin_load(void *cubin);
      static void defer_function_load(void **fat_bin, const char *host_fun,
                                      const char *device_fun);
      static void defer_variable_load(void **fat_bin, char *host_var,
                                      const char *device_name,
                                      bool ext, int size,
                                      bool constant, bool global);
#endif
    public:
      // Helper methods for intercepting CUDA calls
      static GPUProcessor* find_local_gpu(void);
      static void** register_fat_binary(void *fat_bin);
      static void** register_cuda_binary(void *cubin, size_t cubinSize);
      static void unregister_fat_binary(void **fat_bin);
      static void register_var(void **fat_bin, char *host_var,
                               char *device_addr, const char *device_name,
                               int ext, int size, int constant, int global);
      static void register_function(void **fat_bin, const char *host_fun,
                                    char *device_fun, const char *device_name,
                                    int thread_limit, uint3 *tid, uint3 *bid,
                                    dim3 *bDim, dim3 *gDim, int *wSize);
      static char init_module(void **fat_bin);
    public:
      // Helper methods for replacing CUDA calls
#if 0
      static cudaError_t stream_create(cudaStream_t *stream);
      static cudaError_t stream_destroy(cudaStream_t stream);
      static cudaError_t configure_call(dim3 grid_dim, dim3 block_dim,
                                        size_t shared_memory, cudaStream_t stream);
      static cudaError_t setup_argument(const void *arg, size_t size, size_t offset);
      static cudaError_t launch(const void *func);
      static cudaError_t set_shared_memory_config(cudaSharedMemConfig config);
      static const char* get_error_string(cudaError_t error);
      static cudaError_t get_device(int *device);
      static cudaError_t get_device_properties(cudaDeviceProp *prop, int device);
      static cudaError_t get_func_attributes(cudaFuncAttributes *attr, const void *func);
#endif
    };

    class GPUFBMemory : public MemoryImpl {
    public:
      GPUFBMemory(Memory _me, GPU *_gpu, char *_base, size_t _size);

      virtual ~GPUFBMemory(void);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
                                             const Realm::ProfilingRequestSet &reqs,
					     RegionInstance parent_inst);

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      // these work, but they are SLOW
      virtual void get_bytes(off_t offset, void *dst, size_t size);
      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      virtual int get_home_node(off_t offset, size_t size);

    public:
      GPU *gpu;
      char *base;
    };

    class GPUZCMemory : public MemoryImpl {
    public:
      GPUZCMemory(Memory _me, GPU *_gpu, char *_gpu_base, char *_cpu_base, size_t _size);

      virtual ~GPUZCMemory(void);

      virtual RegionInstance create_instance(IndexSpace is,
					     const int *linearization_bits,
					     size_t bytes_needed,
					     size_t block_size,
					     size_t element_size,
					     const std::vector<size_t>& field_sizes,
					     ReductionOpID redopid,
					     off_t list_size,
                                             const Realm::ProfilingRequestSet &reqs,
					     RegionInstance parent_inst);

      virtual void destroy_instance(RegionInstance i, 
				    bool local_destroy);

      virtual off_t alloc_bytes(size_t size);

      virtual void free_bytes(off_t offset, size_t size);

      virtual void get_bytes(off_t offset, void *dst, size_t size);

      virtual void put_bytes(off_t offset, const void *src, size_t size);

      virtual void *get_direct_ptr(off_t offset, size_t size);

      virtual int get_home_node(off_t offset, size_t size);

    public:
      GPU *gpu;
      char *gpu_base;
      char *cpu_base;
    };

  }; // namespace LowLevel
}; // namespace LegionRuntime

#endif
