// Our implementation of the CUDA runtime API for Legion
// so we can intercept all of these calls

// All these extern C methods are for internal implementations
// of functions of the cuda runtime API that nvcc assumes
// exists and can be used for code generation. They are all
// pretty simple to map to the driver API.

// NOTE: This file must only include CUDA runtime API calls and any helpers used
//  ONLY by those calls - there may be NO references to symbols defined in this file
//  by any other parts of Realm, or it will mess up the ability to let an app link
//  against the real libcudart.so

// explicitly include cuda runtime, enabling deprecated things because we're
//  implementing whatever exists, even if it is deprecated - this must precede
//  any transitive inclusion of cuda_runtime.h below
#define CUDA_ENABLE_DEPRECATED
#include <cuda_runtime.h>

#include "realm/cuda/cudart_hijack.h"
#include "realm/cuda/cuda_module.h"
#include "realm/cuda/cuda_internal.h"
#include "realm/logging.h"

namespace Realm {
  namespace Cuda {

    extern Logger log_cudart;
    extern Logger log_gpu;

    struct CallConfig {
      dim3 grid;
      dim3 block;
      size_t shared;
      CUstream stream;
    };
    namespace ThreadLocal {
      extern REALM_THREAD_LOCAL GPUStream *current_gpu_stream;
      extern REALM_THREAD_LOCAL bool block_on_synchronize;
      extern REALM_THREAD_LOCAL std::set<GPUStream *> *created_gpu_streams;
      static thread_local std::vector<CallConfig> call_config_stack;
    } // namespace ThreadLocal

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

    GPUPreemptionWaiter::GPUPreemptionWaiter(GPU *g)
      : gpu(g)
    {
      GenEventImpl *impl = GenEventImpl::create_genevent();
      wait_event = impl->current_event();
    }

    void GPUPreemptionWaiter::request_completed(void)
    {
      GenEventImpl::trigger(wait_event, false /*poisoned*/);
    }

    void GPUPreemptionWaiter::preempt(void)
    {
      // Realm threads don't obey a stack discipline for
      // preemption so we can't leave our context on the stack
      gpu->pop_context();
      wait_event.wait();
      // When we wake back up, we have to push our context again
      gpu->push_context();
    }

    static CUstream translate_stream(CUstream stream)
    {
      if(IS_DEFAULT_STREAM(stream)) {
        return ThreadLocal::current_gpu_stream->get_stream();
      }
      return stream;
    }

    static GPUStream *get_stream_or_die(const char *funcname)
    {
      // mark that the hijack code is active - this covers the calls below
      cudart_hijack_active = true;

      GPUStream *s = ThreadLocal::current_gpu_stream;
      if(s == nullptr) {
        log_cudart.fatal() << funcname << "() called outside CUDA task";
        assert(false);
      }
      return s;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // struct RegisteredFunction

    RegisteredFunction::RegisteredFunction(const FatBin *_fat_bin, const void *_host_fun,
                                           const char *_device_fun)
      : fat_bin(_fat_bin)
      , host_fun(_host_fun)
      , device_fun(_device_fun)
    {}

    ////////////////////////////////////////////////////////////////////////
    //
    // struct RegisteredVariable

    RegisteredVariable::RegisteredVariable(const FatBin *_fat_bin, const void *_host_var,
                                           const char *_device_name, bool _external,
                                           int _size, bool _constant, bool _global,
                                           bool _managed)
      : fat_bin(_fat_bin)
      , host_var(_host_var)
      , device_name(_device_name)
      , external(_external)
      , size(_size)
      , constant(_constant)
      , global(_global)
      , managed(_managed)
    {}

    ////////////////////////////////////////////////////////////////////////
    //
    // class GlobalRegistrations

    GlobalRegistrations::GlobalRegistrations(void) {}

    GlobalRegistrations::~GlobalRegistrations(void)
    {
      variables.clear();
      functions.clear();
      modules.clear();
    }

    /*static*/ GlobalRegistrations &GlobalRegistrations::get_global_registrations(void)
    {
      static GlobalRegistrations reg;
      return reg;
    }

    // called by a GPU when it has created its context - will result in calls back
    //  into the GPU for any modules/variables/whatever already registered
    /*static*/ void GlobalRegistrations::add_gpu_context(GPU *gpu)
    {
      GlobalRegistrations &g = get_global_registrations();

      RWLock::AutoWriterLock al(g.rwlock);

      // add this gpu to the list
      assert(g.active_gpus.count(gpu) == 0);

      for(auto &[fat_bin, mod] : g.modules) {
        g.load_module_under_lock(mod, gpu);
      }

      for(auto &[sym, var] : g.variables) {
        g.register_variable_under_lock(var, gpu);
      }
      for(auto &[sym, func] : g.functions) {
        g.register_function_under_lock(func, gpu);
      }

      g.active_gpus.insert(gpu);
    }

    /*static*/ void GlobalRegistrations::remove_gpu_context(GPU *gpu)
    {
      GlobalRegistrations &g = get_global_registrations();

      RWLock::AutoWriterLock al(g.rwlock);

      assert(g.active_gpus.count(gpu) > 0);
      g.active_gpus.erase(gpu);
    }

    // called by __cuda(un)RegisterFatBinary
    /*static*/ void GlobalRegistrations::register_fat_binary(const FatBin *fat_bin)
    {
      GlobalRegistrations &g = get_global_registrations();

      RWLock::AutoWriterLock al(g.rwlock);

      if(g.modules.find(fat_bin) != g.modules.end()) {
        return;
      }

      RegisteredModule mod;
      mod.fat_bin = fat_bin;

      for(GPU *gpu : g.active_gpus) {
        g.load_module_under_lock(mod, gpu);
      }

      g.modules.insert(std::make_pair(fat_bin, mod));
    }

    /*static*/ void GlobalRegistrations::unregister_fat_binary(const FatBin *fat_bin)
    {
      GlobalRegistrations &g = get_global_registrations();

      RWLock::AutoWriterLock al(g.rwlock);
      // Shouldn't we unload the module from all the gpus?
      g.modules.erase(fat_bin);
    }

    // called by __cudaRegisterVar
    /*static*/ void GlobalRegistrations::register_variable(const RegisteredVariable &v)
    {
      GlobalRegistrations &g = get_global_registrations();

      RWLock::AutoWriterLock al(g.rwlock);
      RegisteredVariable &var = (g.variables[v.host_var] = v);
      assert(g.modules.count(var.fat_bin) > 0);
      for(GPU *gpu : g.active_gpus) {
        g.register_variable_under_lock(var, gpu);
      }
    }

    // called by __cudaRegisterFunction
    /*static*/ void GlobalRegistrations::register_function(const RegisteredFunction &f)
    {
      GlobalRegistrations &g = get_global_registrations();

      RWLock::AutoWriterLock al(g.rwlock);
      RegisteredFunction &func = (g.functions[f.host_fun] = f);
      assert(g.modules.count(func.fat_bin) > 0);
      for(GPU *gpu : g.active_gpus) {
        g.register_function_under_lock(func, gpu);
      }
    }

    /*static*/ CUfunc_st *GlobalRegistrations::lookup_function(const void *func, GPU *gpu)
    {
      GlobalRegistrations &g = get_global_registrations();
      RWLock::AutoReaderLock al(g.rwlock);
      RegisteredFunction &rf = g.functions[func];
      assert(static_cast<size_t>(gpu->info->index) < rf.gpu_functions.size());
      return rf.gpu_functions[gpu->info->index];
    }

    /*static*/ uintptr_t GlobalRegistrations::lookup_variable(const void *var, GPU *gpu)
    {
      GlobalRegistrations &g = get_global_registrations();
      RWLock::AutoReaderLock al(g.rwlock);
      RegisteredVariable &rv = g.variables[var];
      assert(static_cast<size_t>(gpu->info->index) < rv.gpu_addresses.size());
      return rv.gpu_addresses[gpu->info->index];
    }

    void GlobalRegistrations::register_variable_under_lock(RegisteredVariable &var,
                                                           GPU *gpu)
    {
      CUresult result = CUDA_SUCCESS;
      RegisteredModule &mod = modules[var.fat_bin];

      if(var.gpu_addresses.size() <= static_cast<size_t>(gpu->info->index)) {
        var.gpu_addresses.resize(gpu->info->index + 1);
      }
      AutoGPUContext agc(gpu);
      size_t size = 0;
      CUdeviceptr ptr = 0;
      // TODO(cperry): error handling
      result = CUDA_DRIVER_FNPTR(cuModuleGetGlobal)(
          &ptr, &size, mod.gpu_modules[gpu->info->index], var.device_name);
      assert(result == CUDA_SUCCESS);
      var.gpu_addresses[gpu->info->index] = ptr;
      // if this is a managed variable, the "host_var" is actually a pointer
      //  we need to fill in, so do that now
      if(var.managed) {
        CUdeviceptr *indirect =
            const_cast<CUdeviceptr *>(static_cast<const CUdeviceptr *>(var.host_var));
        if((*indirect != 0) && (*indirect != ptr)) {
          log_cudart.fatal() << "__managed__ variables are not supported when using "
                                "multiple devices with CUDART hijack enabled";
          abort();
        }
        *indirect = ptr;
      }
    }

    void GlobalRegistrations::register_function_under_lock(RegisteredFunction &func,
                                                           GPU *gpu)
    {
      CUresult result = CUDA_SUCCESS;
      RegisteredModule &mod = modules[func.fat_bin];

      if(func.gpu_functions.size() <= static_cast<size_t>(gpu->info->index)) {
        func.gpu_functions.resize(gpu->info->index + 1);
      }
      AutoGPUContext agc(gpu);
      // TODO(cperry): error handling
      result = CUDA_DRIVER_FNPTR(cuModuleGetFunction)(
          &func.gpu_functions[gpu->info->index], mod.gpu_modules[gpu->info->index],
          func.device_fun);
      assert(result == CUDA_SUCCESS);
    }

    void GlobalRegistrations::load_module_under_lock(RegisteredModule &mod, GPU *gpu)
    {
      CUresult result = CUDA_SUCCESS;
      const size_t buffer_size = 16384;
      static char log_info_buffer[buffer_size];
      static char log_error_buffer[buffer_size];
      CUjit_option jit_options[] = {
          CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
          CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
      void *option_vals[] = {log_info_buffer, (void *)buffer_size, log_error_buffer,
                             (void *)buffer_size};
      static_assert((sizeof(jit_options) / sizeof(jit_options[0])) ==
                    (sizeof(option_vals) / sizeof(option_vals[0])));

      if(mod.gpu_modules.size() <= static_cast<size_t>(gpu->info->index)) {
        mod.gpu_modules.resize(gpu->info->index + 1);
      }
      // TODO(cperry): error handling
      AutoGPUContext agc(gpu);
      result = CUDA_DRIVER_FNPTR(cuModuleLoadDataEx)(
          &mod.gpu_modules[gpu->info->index], mod.fat_bin->data,
          sizeof(jit_options) / sizeof(jit_options[0]), jit_options, option_vals);
      assert(result == CUDA_SUCCESS);
    }
  }; // namespace Cuda
}; // namespace Realm

////////////////////////////////////////////////////////////////////////
//
// CUDA Runtime API

// these are all "C" functions
extern "C" {

  using namespace Realm;
  using namespace Realm::Cuda;

  REALM_PUBLIC_API
  void** __cudaRegisterFatBinary(const void *fat_bin)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;

//define DEBUG_CUDART_REGISTRATION
#ifdef DEBUG_CUDART_REGISTRATION
    // can't use logger here - this happens at startup
    std::cout << "registering fat binary " << fat_bin << std::endl;
#endif
    GlobalRegistrations::register_fat_binary(reinterpret_cast<const FatBin *>(fat_bin));
    return reinterpret_cast<void **>(const_cast<void *>(fat_bin));
  }

  // This method was added with CUDA 10, not sure what it does yet
  REALM_PUBLIC_API
  void __cudaRegisterFatBinaryEnd(void **fat_bin)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;
    // For now we don't actually do anything in here
    // since it's not immediately obvious what else
    // we might need to do for this cubin since we
    // already registered it using the method above
  }

  REALM_PUBLIC_API
  void __cudaUnregisterFatBinary(void **handle)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;

#ifdef DEBUG_CUDART_REGISTRATION
    std::cout << "unregistering fat binary handle = " << handle << std::endl;
#endif
    GlobalRegistrations::unregister_fat_binary(reinterpret_cast<const FatBin *>(handle));
  }

  REALM_PUBLIC_API
  void __cudaRegisterVar(void **handle,
			 const void *host_var,
			 char *device_addr,
			 const char *device_name,
			 int ext, int size, int constant, int global)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;

#ifdef DEBUG_CUDART_REGISTRATION
    std::cout << "registering variable " << device_name << std::endl;
#endif
    const FatBin *fat_bin = reinterpret_cast<const FatBin *>(handle);
    GlobalRegistrations::register_variable(
        RegisteredVariable(fat_bin, host_var, device_name, ext != 0, size, constant != 0,
                           global != 0, false /*unmanaged*/));
  }

  REALM_PUBLIC_API
  void __cudaRegisterManagedVar(void **handle,
                                const void *host_var,
                                char *device_addr,
                                const char *device_name,
                                int ext, int size, int constant, int global)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;

#ifdef DEBUG_CUDART_REGISTRATION
    std::cout << "registering managed variable " << device_name << std::endl;
#endif
    const FatBin *fat_bin = reinterpret_cast<const FatBin *>(handle);
    GlobalRegistrations::register_variable(
        RegisteredVariable(fat_bin, host_var, device_name, ext != 0, size, constant != 0,
                           global != 0, true /*managed*/));
  }
      
  REALM_PUBLIC_API
  void __cudaRegisterFunction(void **handle,
			      const void *host_fun,
			      char *device_fun,
			      const char *device_name,
			      int thread_limit,
			      uint3 *tid, uint3 *bid,
			      dim3 *bDim, dim3 *gDim,
			      int *wSize)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;

#ifdef DEBUG_CUDART_REGISTRATION
    std::cout << "registering function " << device_fun << ", handle = " << handle << std::endl;
#endif
    const FatBin *fat_bin = reinterpret_cast<const FatBin *>(handle);
    GlobalRegistrations::register_function(
        RegisteredFunction(fat_bin, host_fun, device_fun));
  }
      
  REALM_PUBLIC_API
  char __cudaInitModule(void **fat_bin)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;

    // don't care - return 1 to make caller happy
    return 1;
  } 

  // All the following methods are cuda runtime API calls that we 
  // intercept and then either execute using the driver API or 
  // modify in ways that are important to Legion.

  REALM_PUBLIC_API
  unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedSize = 0,
                                       void *stream = nullptr)
  {

    (void)get_stream_or_die(__FUNCTION__);
    Realm::Cuda::ThreadLocal::call_config_stack.push_back(
        {gridDim, blockDim, sharedSize, (CUstream)stream});

    return 0;
  }

  REALM_PUBLIC_API
  cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
					 dim3 *blockDim,
					 size_t *sharedSize,
					 void *stream)
  {
    (void)get_stream_or_die(__FUNCTION__);
    Realm::Cuda::CallConfig &config = Realm::Cuda::ThreadLocal::call_config_stack.back();
    *gridDim = config.grid;
    *blockDim = config.block;
    *sharedSize = config.shared;
    *((CUstream *)stream) = config.stream;

    Realm::Cuda::ThreadLocal::call_config_stack.pop_back();

    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventCreate(cudaEvent_t *event)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuEventCreate)(event, CU_EVENT_DEFAULT);
  }
	
  REALM_PUBLIC_API
  cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
				       unsigned int flags)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuEventCreate)(event, flags);
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream /*= 0*/)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuEventRecord)(event, translate_stream(stream));
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventSynchronize(cudaEvent_t event)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuEventSynchronize)(event);
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventDestroy(cudaEvent_t event)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuEventDestroy)(event);
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuEventElapsedTime)(ms, start, end);
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamCreate(cudaStream_t *stream)
  {
    // This needs to be a blocking stream that serializes with the
    // "NULL" stream, which in this case is the original stream
    // for the task, so the only way to enforce that currently is 
    // with exactly the same stream
    *stream = get_stream_or_die(__FUNCTION__)->get_stream();
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    if(flags == cudaStreamNonBlocking) {
      *stream = s->get_gpu()->get_next_task_stream(true)->get_stream();
    } else {
      *stream = s->get_stream();
    }
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, 
					   unsigned int flags, int priority)
  {
    // We'll ignore the priority for now
    return cudaStreamCreateWithFlags(stream, flags);
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamDestroy(cudaStream_t stream)
  {
    /*GPUProcessor *p =*/get_stream_or_die(__FUNCTION__);
    // Ignore this for now because we know we can reuse streams as needed
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamSynchronize(cudaStream_t stream)
  {
    GPUStream *current = get_stream_or_die(__FUNCTION__);
    if(IS_DEFAULT_STREAM(stream)) {
      return cudaDeviceSynchronize();
    }
    if(Realm::Cuda::ThreadLocal::block_on_synchronize) {
      return (cudaError_t)CUDA_DRIVER_FNPTR(cuStreamSynchronize)(stream);
    }
    GPUStream *s = current->get_gpu()->find_stream(stream);
    if(s == nullptr) {
      log_gpu.warning() << "WARNING: Detected unknown CUDA stream " << stream
                        << " that Realm did not create which suggests "
                        << "that there is another copy of the CUDA runtime "
                        << "somewhere making its own streams... be VERY careful.";
      return (cudaError_t)CUDA_DRIVER_FNPTR(cuStreamSynchronize)(stream);
    }

    // We don't actually want to block the GPU processor
    // when synchronizing, so we instead register a cuda
    // event on the stream and then use it triggering to
    // indicate that the stream is caught up
    // Make a completion notification to be notified when
    // the event has actually triggered
    GPUPreemptionWaiter waiter(current->get_gpu());
    // Register the waiter with the stream
    s->add_notification(&waiter);
    // Perform the wait, this will preempt the thread
    waiter.preempt();

    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
				  cudaEvent_t event,
				  unsigned int flags)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuStreamWaitEvent)(translate_stream(stream),
                                                             event, flags);
  }

  REALM_PUBLIC_API
  cudaError_t cudaLaunchKernel(const void *func,
			       dim3 grid_dim,
			       dim3 block_dim,
			       void **args,
			       size_t shared_memory,
			       cudaStream_t stream)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    CUfunction f = GlobalRegistrations::lookup_function(func, s->get_gpu());
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuLaunchKernel)(
        f, grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z,
        shared_memory, translate_stream(stream), args, nullptr);
  }

#if CUDA_VERSION >= 9000
  REALM_PUBLIC_API
  cudaError_t cudaLaunchCooperativeKernel(const void *func,
					  dim3 grid_dim,
					  dim3 block_dim,
					  void **args,
					  size_t shared_memory,
					  cudaStream_t stream)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    CUfunction f = GlobalRegistrations::lookup_function(func, s->get_gpu());
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuLaunchCooperativeKernel)(
        f, grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z,
        shared_memory, translate_stream(stream), args);
  }
#endif

  REALM_PUBLIC_API
  cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemGetInfo)(free, total);
  }

  REALM_PUBLIC_API
  cudaError_t cudaMalloc(void **ptr, size_t size)
  {
    (void)get_stream_or_die(__FUNCTION__);

    // CUDART tolerates size=0, returning nullptr, but the driver API
    //  does not, so catch that here
    if(size == 0) {
      *ptr = 0;
      return cudaSuccess;
    }

    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemAlloc)(
        reinterpret_cast<CUdeviceptr *>(ptr), size);
  }

  REALM_PUBLIC_API
  cudaError_t cudaMallocHost(void **ptr, size_t size)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return cudaHostAlloc(ptr, size, 0);
  }

  REALM_PUBLIC_API
  cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
  {
    (void)get_stream_or_die(__FUNCTION__);

    // CUDART tolerates size=0, returning nullptr, but the driver API
    //  does not, so catch that here
    if(size == 0) {
      *ptr = 0;
      return cudaSuccess;
    }

    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemHostAlloc)(ptr, size, flags);
  }

#if CUDART_VERSION >= 8000
  // Managed memory is only supported for cuda version >=8.0
  REALM_PUBLIC_API
  cudaError_t cudaMallocManaged(void **ptr, size_t size,
				unsigned flags /*= cudaMemAttachGlobal*/)
  {
    (void)get_stream_or_die(__FUNCTION__);

    // CUDART tolerates size=0, returning nullptr, but the driver API
    //  does not, so catch that here
    if(size == 0) {
      *ptr = 0;
      return cudaSuccess;
    }

    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemAllocManaged)(
        reinterpret_cast<CUdeviceptr *>(ptr), size, flags);
  }

  REALM_PUBLIC_API
  cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attr,
				       const void *ptr)
  {
    CUresult ret = CUDA_SUCCESS;
    (void)get_stream_or_die(__FUNCTION__);

    unsigned mem_type = 0;
    CUdeviceptr dptr = 0;
    void *hptr = nullptr;
    int dev = 0;

    CUpointer_attribute attrs[] = {
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
        CU_POINTER_ATTRIBUTE_HOST_POINTER, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};

    void *data[] = {&mem_type, &dptr, &hptr, &dev};

    if(CUDA_SUCCESS != (ret = CUDA_DRIVER_FNPTR(cuPointerGetAttributes)(
                            sizeof(attrs) / sizeof(attrs[0]), attrs, data,
                            reinterpret_cast<CUdeviceptr>(ptr)))) {
      if(ret == CUDA_ERROR_INVALID_VALUE) {
        attr->type = cudaMemoryTypeUnregistered;
        attr->device = 0;
        attr->devicePointer = 0;
        attr->hostPointer = const_cast<void *>(ptr);
        return cudaSuccess;
      }
      return (cudaError_t)ret;
    }

    attr->device = dev;
    attr->devicePointer = reinterpret_cast<void *>(dptr);
    attr->hostPointer = hptr;
    attr->type = (cudaMemoryType)mem_type;

    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemPrefetchAsync(const void *dev_ptr,
				   size_t count,
				   int dst_device,
				   cudaStream_t stream)
  {
    (void)get_stream_or_die(__FUNCTION__);

    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemPrefetchAsync)(
        (CUdeviceptr)dev_ptr, count, dst_device, translate_stream(stream));
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemAdvise(const void *dev_ptr,
                            size_t count,
                            cudaMemoryAdvise advice,
                            int device)
  {
    (void)get_stream_or_die(__FUNCTION__);

    // NOTE: we assume the enums for cudaMeoryAdvise match the ones for
    //  CUmem_advise
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemAdvise)(
        (CUdeviceptr)dev_ptr, count, (CUmem_advise)advice, (CUdevice)device);
  }
#endif

  REALM_PUBLIC_API
  cudaError_t cudaFree(void *ptr)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemFree)((CUdeviceptr)ptr);
  }

  REALM_PUBLIC_API
  cudaError_t cudaFreeHost(void *ptr)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemFree)((CUdeviceptr)ptr);
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpy(void *dst, const void *src, 
			 size_t size, cudaMemcpyKind kind)
  {
    cudaError_t ret = cudaSuccess;
    ret = cudaMemcpyAsync(dst, src, size, kind, 0);
    if(ret == cudaSuccess) {
      ret = cudaStreamSynchronize(0);
    }
    return ret;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpyAsync(void *dst, const void *src,
			      size_t size, cudaMemcpyKind kind,
			      cudaStream_t stream)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemcpyAsync)(
        (CUdeviceptr)dst, (CUdeviceptr)src, size, translate_stream(stream));
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			   size_t width, size_t height, cudaMemcpyKind kind)
  {
    cudaError_t ret = cudaSuccess;
    ret = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, 0);
    if(ret == cudaSuccess) {
      ret = cudaStreamSynchronize(0);
    }
    return ret;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, 
				size_t spitch, size_t width, size_t height, 
				cudaMemcpyKind kind, cudaStream_t stream)
  {
    (void)get_stream_or_die(__FUNCTION__);
    CUDA_MEMCPY2D copy_info;
    copy_info.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy_info.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy_info.srcDevice = (CUdeviceptr)src;
    copy_info.srcHost = src;
    copy_info.srcPitch = spitch;
    copy_info.srcY = 0;
    copy_info.srcXInBytes = 0;
    copy_info.dstDevice = (CUdeviceptr)dst;
    copy_info.dstHost = dst;
    copy_info.dstPitch = dpitch;
    copy_info.dstY = 0;
    copy_info.dstXInBytes = 0;
    copy_info.WidthInBytes = width;
    copy_info.Height = height;
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemcpy2DAsync)(&copy_info,
                                                           translate_stream(stream));
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemset(void *dst, int value, size_t count)
  {
    cudaError_t ret = cudaSuccess;
    ret = cudaMemsetAsync(dst, value, count, 0);
    if(ret == cudaSuccess) {
      ret = cudaStreamSynchronize(0);
    }
    return ret;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemsetAsync(void *dst, int value, 
			      size_t count, cudaStream_t stream)
  {
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemsetD8Async)(
        (CUdeviceptr)dst, (unsigned char)value, count, translate_stream(stream));
  }

  REALM_PUBLIC_API
  cudaError_t cudaDeviceSynchronize(void)
  {
    cudart_hijack_active = true;

    GPUStream *current = Realm::Cuda::ThreadLocal::current_gpu_stream;
    if(current != nullptr) {
      if(Realm::Cuda::ThreadLocal::created_gpu_streams != nullptr) {
        current->wait_on_streams(*Realm::Cuda::ThreadLocal::created_gpu_streams);
        delete Realm::Cuda::ThreadLocal::created_gpu_streams;
        Realm::Cuda::ThreadLocal::created_gpu_streams = 0;
      }
      return cudaStreamSynchronize(current->get_stream());
    }

    switch(cudart_hijack_nongpu_sync) {
    case 2:
    default:
      log_cudart.fatal() << "cudaDeviceSynchronize() called outside CUDA task";
      abort();
      break;
    case 1: // warn
    {
      Backtrace bt;
      bt.capture_backtrace();
      bt.lookup_symbols();
      log_cudart.warning() << "cudaDeviceSynchronize() called outside CUDA task at "
                           << bt;
      break;
    }
    case 0: // ignore
      break;
    }

    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpyToSymbol(const void *dst, const void *src,
				 size_t size, size_t offset,
				 cudaMemcpyKind kind)
  {
    cudaError_t ret = cudaMemcpyToSymbolAsync(dst, src, size, offset, kind, 0);
    if(ret == cudaSuccess) {
      ret = cudaStreamSynchronize(0);
    }
    return ret;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpyToSymbolAsync(const void *dst, const void *src,
				      size_t size, size_t offset,
				      cudaMemcpyKind kind, cudaStream_t stream)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemcpyAsync)(
        GlobalRegistrations::lookup_variable(dst, s->get_gpu()) + offset,
        (CUdeviceptr)src, size, translate_stream(stream));
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpyFromSymbol(void *dst, const void *src,
				   size_t size, size_t offset,
				   cudaMemcpyKind kind)
  {
    cudaError_t ret = cudaMemcpyFromSymbolAsync(dst, src, size, offset, kind, 0);
    if(ret == cudaSuccess) {
      ret = cudaStreamSynchronize(0);
    }
    return ret;
  }
      
  REALM_PUBLIC_API
  cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *src,
					size_t size, size_t offset,
					cudaMemcpyKind kind, cudaStream_t stream)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuMemcpyAsync)(
        (CUdeviceptr)dst,
        GlobalRegistrations::lookup_variable(src, s->get_gpu()) + offset, size,
        translate_stream(stream));
  }
      
  REALM_PUBLIC_API
  cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config)
  {

    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuCtxSetSharedMemConfig)(
        (CUsharedconfig)config);
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetDevice(int *device)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    *device = s->get_gpu()->info->index;
    return cudaSuccess;
  }

#define CUDA_DEVICE_ATTRIBUTES_PRE_8_0(__op__) \
  __op__(major, COMPUTE_CAPABILITY_MAJOR)                  \
  __op__(minor, COMPUTE_CAPABILITY_MINOR)                  \
  __op__(sharedMemPerBlock, MAX_SHARED_MEMORY_PER_BLOCK)   \
  __op__(regsPerBlock,      MAX_REGISTERS_PER_BLOCK)       \
  __op__(warpSize,          WARP_SIZE)                     \
  __op__(memPitch,          MAX_PITCH)                     \
  __op__(maxThreadsPerBlock, MAX_THREADS_PER_BLOCK)        \
  __op__(maxThreadsDim[0], MAX_BLOCK_DIM_X)                \
  __op__(maxThreadsDim[1], MAX_BLOCK_DIM_Y)                \
  __op__(maxThreadsDim[2], MAX_BLOCK_DIM_Z)                \
  __op__(maxGridSize[0], MAX_GRID_DIM_X)                   \
  __op__(maxGridSize[1], MAX_GRID_DIM_Y)                   \
  __op__(maxGridSize[2], MAX_GRID_DIM_Z)                   \
  __op__(maxSurface1D,  MAXIMUM_SURFACE1D_WIDTH)           \
  __op__(maxSurface2D[0], MAXIMUM_SURFACE2D_WIDTH)         \
  __op__(maxSurface2D[1], MAXIMUM_SURFACE2D_HEIGHT)        \
  __op__(maxSurface3D[0], MAXIMUM_SURFACE3D_WIDTH)         \
  __op__(maxSurface3D[1], MAXIMUM_SURFACE3D_HEIGHT)        \
  __op__(maxSurface3D[2], MAXIMUM_SURFACE3D_DEPTH)         \
  __op__(maxSurface1DLayered[0],  MAXIMUM_SURFACE1D_LAYERED_WIDTH)           \
  __op__(maxSurface1DLayered[1],  MAXIMUM_SURFACE1D_LAYERED_LAYERS)          \
  __op__(maxSurface2DLayered[0], MAXIMUM_SURFACE2D_LAYERED_WIDTH)            \
  __op__(maxSurface2DLayered[1], MAXIMUM_SURFACE2D_LAYERED_HEIGHT)           \
  __op__(maxSurface2DLayered[2], MAXIMUM_SURFACE2D_LAYERED_LAYERS)           \
  __op__(maxSurfaceCubemap, MAXIMUM_SURFACECUBEMAP_WIDTH)                    \
  __op__(maxSurfaceCubemapLayered[0], MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH)  \
  __op__(maxSurfaceCubemapLayered[1], MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS) \
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

  REALM_PUBLIC_API
  cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int index)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    const std::vector<GPUInfo *> &infos = s->get_gpu()->module->gpu_info;
    for (GPUInfo* info : infos) {
      if (info->index != index)
        continue;

      if (info->prop.totalGlobalMem != info->totalGlobalMem) {
        // TODO: Replace with cuDeviceGetAttributes to batch
#define CUDA_GET_DEVICE_PROP(name, attr)                                                 \
  do {                                                                                   \
    int val = 0;                                                                         \
    (void)CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(&val, CU_DEVICE_ATTRIBUTE_##attr,      \
                                                  info->device);                         \
    info->prop.name = val;                                                               \
  } while(0);
        CUDA_DEVICE_ATTRIBUTES(CUDA_GET_DEVICE_PROP);

        strncpy(info->prop.name, info->name, sizeof(info->prop.name));
        info->prop.totalGlobalMem = info->totalGlobalMem;
        #undef CUDA_GET_DEVICE_PROP
      }

      memcpy(prop, &info->prop, sizeof(*prop));

      return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
  }
      
  REALM_PUBLIC_API
  cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int index)
  {
    CUresult ret = CUDA_SUCCESS;
    CUdevice device;
    ret = CUDA_DRIVER_FNPTR(cuDeviceGet)(&device, index);
    if(ret == CUDA_SUCCESS) {
      ret = CUDA_DRIVER_FNPTR(cuDeviceGetAttribute)(value, (CUdevice_attribute)attr,
                                                    device);
    }
    return (cudaError_t)ret;
  }

  REALM_PUBLIC_API
  cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, const void *func)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);

    CUfunction handle = GlobalRegistrations::lookup_function(func, s->get_gpu());

#define GET_FUNC_ATTR(member, name)                                                      \
  do {                                                                                   \
    int tmp;                                                                             \
    (void)CUDA_DRIVER_FNPTR(cuFuncGetAttribute)(&tmp, CU_FUNC_ATTRIBUTE_##name, handle); \
    attr->member = tmp;                                                                  \
  } while(0)

    GET_FUNC_ATTR(binaryVersion, BINARY_VERSION);
    GET_FUNC_ATTR(cacheModeCA, CACHE_MODE_CA);
    GET_FUNC_ATTR(constSizeBytes, CONST_SIZE_BYTES);
    GET_FUNC_ATTR(localSizeBytes, LOCAL_SIZE_BYTES);
#if CUDA_VERSION >= 9000
    GET_FUNC_ATTR(maxDynamicSharedSizeBytes, MAX_DYNAMIC_SHARED_SIZE_BYTES);
#endif
    GET_FUNC_ATTR(maxThreadsPerBlock, MAX_THREADS_PER_BLOCK);
    GET_FUNC_ATTR(numRegs, NUM_REGS);
#if CUDA_VERSION >= 9000
    GET_FUNC_ATTR(preferredShmemCarveout, PREFERRED_SHARED_MEMORY_CARVEOUT);
#endif
    GET_FUNC_ATTR(ptxVersion, PTX_VERSION);
    GET_FUNC_ATTR(sharedSizeBytes, SHARED_SIZE_BYTES);
#undef GET_FUNC_ATTR

    return cudaSuccess;
  }

#if CUDA_VERSION >= 9000
  REALM_PUBLIC_API
  cudaError_t cudaFuncSetAttribute(const void *func, cudaFuncAttribute attr, int value)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);

    CUfunction handle = GlobalRegistrations::lookup_function(func, s->get_gpu());
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuFuncSetAttribute)(
        handle, (CUfunction_attribute)attr, value);
  }
#endif

  REALM_PUBLIC_API
  cudaError_t cudaFuncSetCacheConfig(const void *func, cudaFuncCache cacheConfig)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);

    CUfunction handle = GlobalRegistrations::lookup_function(func, s->get_gpu());
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuFuncSetCacheConfig)(
        handle, (CUfunc_cache)cacheConfig);
  }

  REALM_PUBLIC_API
  cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache config)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuCtxSetCacheConfig)((CUfunc_cache)config);
  }

  REALM_PUBLIC_API
  cudaError_t cudaFuncSetSharedMemConfig(const void *func, cudaSharedMemConfig config)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    CUfunction handle = GlobalRegistrations::lookup_function(func, s->get_gpu());
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuFuncSetSharedMemConfig)(
        handle, (CUsharedconfig)config);
  }

#if CUDA_VERSION >= 11000
  REALM_PUBLIC_API
  cudaError_t cudaGetFuncBySymbol(cudaFunction_t *funcPtr, const void *symbolPtr)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    *funcPtr = GlobalRegistrations::lookup_function(symbolPtr, s->get_gpu());
    return cudaSuccess;
  }
#endif

  REALM_PUBLIC_API
  cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
							    const void *func,
							    int blockSize,
							    size_t dynamicSMemSize)
  {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, func, blockSize, dynamicSMemSize, 0);
  }

  REALM_PUBLIC_API
  cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, 
								     const void *func, 
								     int blockSize, 
								     size_t dynamicSMemSize,
								     unsigned int flags)
  {
    GPUStream *s = get_stream_or_die(__FUNCTION__);
    CUfunction handle = GlobalRegistrations::lookup_function(func, s->get_gpu());
    return (cudaError_t)CUDA_DRIVER_FNPTR(
        cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(
        numBlocks, handle, blockSize, dynamicSMemSize, flags);
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetLastError(void)
  {
    get_stream_or_die(__FUNCTION__);
    // For now we're not tracking this so if we were
    // going to die we already would have
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaPeekAtLastError(void)
  {
    get_stream_or_die(__FUNCTION__);
    // For now we're not tracking this so if we were
    // going to die we already would have
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  const char* cudaGetErrorName(cudaError_t error)
  {
    static const char unknown_str[] = "Unknown";
    const char *result = nullptr;
    if(CUDA_DRIVER_FNPTR(cuGetErrorName)((CUresult)error, &result) != CUDA_SUCCESS) {
      result = unknown_str;
    }
    return result;
  }

  REALM_PUBLIC_API
  const char* cudaGetErrorString(cudaError_t error)
  {
    static const char unknown_str[] = "Unknown";
    const char *result = nullptr;
    if(CUDA_DRIVER_FNPTR(cuGetErrorString)((CUresult)error, &result) != CUDA_SUCCESS) {
      result = unknown_str;
    }
    return result;
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetDeviceCount(int *count)
  {
    // TODO: lie here and report just one device?
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuDeviceGetCount)(count);
  }

  REALM_PUBLIC_API
  cudaError_t cudaSetDevice(int device)
  {
    (void)get_stream_or_die(__FUNCTION__);
    // Ignore calls to set the device here since we already
    // know which device we are running on
    return cudaSuccess;
  } 

  REALM_PUBLIC_API
  cudaError_t cudaThreadSetCacheConfig(cudaFuncCache config)
  {
    // old name for cudaDeviceSetCacheConfig
    return cudaDeviceSetCacheConfig(config);
  }

  REALM_PUBLIC_API
  cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
					      cudaChannelFormatKind f)
  {
    cudaChannelFormatDesc desc;
    desc.x = x;
    desc.y = y;
    desc.z = z;
    desc.w = w;
    desc.f = f;
    return desc;
  }

  REALM_PUBLIC_API
  cudaError_t cudaCreateTextureObject(cudaTextureObject_t *tex,
				      const cudaResourceDesc *res_desc,
				      const cudaTextureDesc *tex_desc,
				      const cudaResourceViewDesc *view_desc)
  {
    // TODO: support
    return cudaErrorInvalidValue;
  }

  REALM_PUBLIC_API
  cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuCtxSetLimit)((CUlimit)limit, value);
  }

  REALM_PUBLIC_API
  cudaError_t cudaDeviceGetLimit(size_t *value, cudaLimit limit)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuCtxGetLimit)(value, (CUlimit)limit);
  }

  REALM_PUBLIC_API
  cudaError_t cudaSetDeviceFlags(unsigned flags)
  {
    (void)get_stream_or_die(__FUNCTION__);

    // flags must be set before contexts are created, which is before any
    //  task can run in Realm, so we know it's too late
    return cudaErrorSetOnActiveProcess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetDeviceFlags(unsigned *flags)
  {
    (void)get_stream_or_die(__FUNCTION__);
    return (cudaError_t)CUDA_DRIVER_FNPTR(cuCtxGetFlags)(flags);
  }

  REALM_PUBLIC_API
  cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                                const cudaChannelFormatDesc *desc,
                                cudaExtent extent, unsigned flags)
  {
    (void)get_stream_or_die(__FUNCTION__);

    CUDA_ARRAY3D_DESCRIPTOR d;
    d.Width = extent.width;
    d.Height = extent.height;
    d.Depth = extent.depth;

    // runtime and driver describe channel count/format very differently
    int channel_count = 0;
    int channel_width = 0;
    if(desc->x != 0) {
      channel_count++;
      channel_width = desc->x;
    }
    if(desc->y != 0) {
      channel_count++;
      if(channel_width && (desc->y != channel_width))
        return cudaErrorInvalidValue;
      channel_width = desc->y;
    }
    if(desc->z != 0) {
      channel_count++;
      if(channel_width && (desc->z != channel_width))
        return cudaErrorInvalidValue;
      channel_width = desc->z;
    }
    if(desc->w != 0) {
      channel_count++;
      if(channel_width && (desc->w != channel_width))
        return cudaErrorInvalidValue;
      channel_width = desc->w;
    }

    switch(desc->f) {
    case cudaChannelFormatKindSigned:
      {
        switch(channel_width) {
        case 8 : { d.Format = CU_AD_FORMAT_SIGNED_INT8; break; }
        case 16 : { d.Format = CU_AD_FORMAT_SIGNED_INT16; break; }
        case 32 : { d.Format = CU_AD_FORMAT_SIGNED_INT32; break; }
        default: return cudaErrorInvalidValue;
        }
        break;
      }
    case cudaChannelFormatKindUnsigned:
      {
        switch(channel_width) {
        case 8 : { d.Format = CU_AD_FORMAT_UNSIGNED_INT8; break; }
        case 16 : { d.Format = CU_AD_FORMAT_UNSIGNED_INT16; break; }
        case 32 : { d.Format = CU_AD_FORMAT_UNSIGNED_INT32; break; }
        default: return cudaErrorInvalidValue;
        }
        break;
      }
    case cudaChannelFormatKindFloat:
      {
        switch(channel_width) {
        case 16 : { d.Format = CU_AD_FORMAT_HALF; break; }
        case 32 : { d.Format = CU_AD_FORMAT_FLOAT; break; }
        default: return cudaErrorInvalidValue;
        }
        break;
      }
    default: return cudaErrorInvalidValue;
    }

    d.NumChannels = channel_count;

    d.Flags = 0;
    if((flags & cudaArrayLayered) != 0)
      d.Flags |= CUDA_ARRAY3D_LAYERED;
    if((flags & cudaArrayCubemap) != 0)
      d.Flags |= CUDA_ARRAY3D_CUBEMAP;
    if((flags & cudaArraySurfaceLoadStore) != 0)
      d.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
    if((flags & cudaArrayTextureGather) != 0)
      d.Flags |= CUDA_ARRAY3D_TEXTURE_GATHER;
    // ignore flags we don't understand and hope for the best

    return (cudaError_t)CUDA_DRIVER_FNPTR(cuArray3DCreate)((CUarray *)array, &d);
  }

  REALM_PUBLIC_API
  cudaError_t cudaFreeArray(cudaArray_t array)
  {
    (void)get_stream_or_die(__FUNCTION__);

    return (cudaError_t)CUDA_DRIVER_FNPTR(cuArrayDestroy)((CUarray)array);
  }

  REALM_PUBLIC_API
  cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *object,
                                      const cudaResourceDesc *desc)
  {
    (void)get_stream_or_die(__FUNCTION__);

    if(desc->resType != cudaResourceTypeArray) {
      return cudaErrorInvalidValue;
    }

    CUDA_RESOURCE_DESC d;
    memset(&d, 0, sizeof(d));
    d.resType = CU_RESOURCE_TYPE_ARRAY;
    d.res.array.hArray = (CUarray)(desc->res.array.array);
    d.flags = 0;

    return (cudaError_t)CUDA_DRIVER_FNPTR(cuSurfObjectCreate)((CUsurfObject *)object, &d);
  }

  REALM_PUBLIC_API
  cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t object)
  {
    (void)get_stream_or_die(__FUNCTION__);

    return (cudaError_t)CUDA_DRIVER_FNPTR(cuSurfObjectDestroy)((CUsurfObject)object);
  }

}; // extern "C"
