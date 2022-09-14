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

    static GPUProcessor *get_gpu_or_die(const char *funcname)
    {
      // mark that the hijack code is active - this covers the calls below
      cudart_hijack_active = true;

      GPUProcessor *p = GPUProcessor::get_current_gpu_proc();
      if (!p) {
        log_cudart.fatal() << funcname << "() called outside CUDA task";
        assert(false);
      }
      return p;
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

    // we make a "handle" that just holds the pointer
    void **handle = new void *;
    *handle = (void *)fat_bin;
//define DEBUG_CUDART_REGISTRATION
#ifdef DEBUG_CUDART_REGISTRATION
    // can't use logger here - this happens at startup
    std::cout << "registering fat binary " << fat_bin << ", handle = " << handle << std::endl;
#endif
    GlobalRegistrations::register_fat_binary((FatBin *)fat_bin);
    return handle;
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

    FatBin *fat_bin = *(FatBin **)handle;
#ifdef DEBUG_CUDART_REGISTRATION
    std::cout << "unregistering fat binary " << fat_bin << ", handle = " << handle << std::endl;
#endif
    GlobalRegistrations::unregister_fat_binary(fat_bin);

    delete handle;
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
    const FatBin *fat_bin = *(const FatBin **)handle;
    GlobalRegistrations::register_variable(new RegisteredVariable(fat_bin,
								  host_var,
								  device_name,
								  ext != 0,
								  size,
								  constant != 0,
								  global != 0,
                                                                  false /*unmanaged*/));
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
    const FatBin *fat_bin = *(const FatBin **)handle;
    GlobalRegistrations::register_variable(new RegisteredVariable(fat_bin,
								  host_var,
								  device_name,
								  ext != 0,
								  size,
								  constant != 0,
								  global != 0,
                                                                  true /*managed*/));
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
    const FatBin *fat_bin = *(const FatBin **)handle;
    GlobalRegistrations::register_function(new RegisteredFunction(fat_bin,
								  host_fun,
								  device_fun));
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
  unsigned __cudaPushCallConfiguration(dim3 gridDim,
				       dim3 blockDim,
				       size_t sharedSize = 0,
				       void *stream = 0)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;

    GPUProcessor *p = get_gpu_or_die("__cudaPushCallConfigration");
    p->push_call_configuration(gridDim, blockDim, sharedSize, stream);
    // This should always succeed so return 0
    return 0;
  }

  REALM_PUBLIC_API
  cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
					 dim3 *blockDim,
					 size_t *sharedSize,
					 void *stream)
  {
    // mark that the hijack code is active
    cudart_hijack_active = true;

    GPUProcessor *p = get_gpu_or_die("__cudaPopCallConfigration");
    p->pop_call_configuration(gridDim, blockDim, sharedSize, stream);
    // If this failed it would have died with an assertion internally
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventCreate(cudaEvent_t *event)
  {
    GPUProcessor *p = get_gpu_or_die("cudaEventCreate");
    p->event_create(event, cudaEventDefault);
    return cudaSuccess;
  }
	
  REALM_PUBLIC_API
  cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
				       unsigned int flags)
  {
    GPUProcessor *p = get_gpu_or_die("cudaEventCreateWithFlags");
    p->event_create(event, flags);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream /*= 0*/)
  {
    GPUProcessor *p = get_gpu_or_die("cudaEventRecord");
    p->event_record(event, stream);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventSynchronize(cudaEvent_t event)
  {
    GPUProcessor *p = get_gpu_or_die("cudaEventSynchronize");
    p->event_synchronize(event);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventDestroy(cudaEvent_t event)
  {
    GPUProcessor *p = get_gpu_or_die("cudaEventDestroy");
    p->event_destroy(event);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
  {
    GPUProcessor *p = get_gpu_or_die("cudaEventElapsedTime");
    p->event_elapsed_time(ms, start, end);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamCreate(cudaStream_t *stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaStreamCreate");
    // This needs to be a blocking stream that serializes with the
    // "NULL" stream, which in this case is the original stream
    // for the task, so the only way to enforce that currently is 
    // with exactly the same stream
    *stream = p->gpu->get_null_task_stream()->get_stream();
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
  {
    GPUProcessor *p = get_gpu_or_die("cudaStreamCreateWithFlags");
    if (flags == cudaStreamNonBlocking)
      *stream = p->gpu->get_next_task_stream(true/*create*/)->get_stream();
    else
      *stream = p->gpu->get_null_task_stream()->get_stream();
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
    /*GPUProcessor *p =*/ get_gpu_or_die("cudaStreamDestroy");
    // Ignore this for now because we know our streams are not being created on the fly
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamSynchronize(cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaStreamSynchronize");
    p->stream_synchronize(stream);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
				  cudaEvent_t event,
				  unsigned int flags)
  {
    GPUProcessor *p = get_gpu_or_die("cudaStreamWaitEvent");
    p->stream_wait_on_event(stream, event);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaConfigureCall(dim3 grid_dim,
				dim3 block_dim,
				size_t shared_memory,
				cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaConfigureCall");
    p->configure_call(grid_dim, block_dim, shared_memory, stream);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaSetupArgument(const void *arg,
				size_t size,
				size_t offset)
  {
    GPUProcessor *p = get_gpu_or_die("cudaSetupArgument");
    p->setup_argument(arg, size, offset);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaLaunch(const void *func)
  {
    GPUProcessor *p = get_gpu_or_die("cudaLaunch");
    p->launch(func);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaLaunchKernel(const void *func,
			       dim3 grid_dim,
			       dim3 block_dim,
			       void **args,
			       size_t shared_memory,
			       cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaLaunchKernel");
    p->launch_kernel(func, grid_dim, block_dim, args, shared_memory, stream);
    return cudaSuccess;
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
    GPUProcessor *p = get_gpu_or_die("cudaLaunchCooperativeKernel");
    p->launch_kernel(func, grid_dim, block_dim, args, 
		     shared_memory, stream, true/*cooperative*/);
    return cudaSuccess;
  }
#endif

  REALM_PUBLIC_API
  cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
  {
    /*GPUProcessor *p =*/ get_gpu_or_die("cudaMemGetInfo");

    CUresult ret = cuMemGetInfo(free, total);
    if(ret == CUDA_SUCCESS) return cudaSuccess;
    assert(ret == CUDA_ERROR_INVALID_VALUE);
    return cudaErrorInvalidValue;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMalloc(void **ptr, size_t size)
  {
    /*GPUProcessor *p =*/ get_gpu_or_die("cudaMalloc");

    // CUDART tolerates size=0, returning nullptr, but the driver API
    //  does not, so catch that here
    if(size == 0) {
      *ptr = 0;
      return cudaSuccess;
    } else {
      CUresult ret = cuMemAlloc((CUdeviceptr *)ptr, size);
      if(ret == CUDA_SUCCESS) return cudaSuccess;
      assert(ret == CUDA_ERROR_OUT_OF_MEMORY);
      return cudaErrorMemoryAllocation;
    }
  }

  REALM_PUBLIC_API
  cudaError_t cudaMallocHost(void **ptr, size_t size)
  {
    get_gpu_or_die("cudaMallocHost");

    // size=0 is allowed, but the runtime doesn't always do a good job of
    //  setting *ptr to nullptr, so do it ourselves
    if(size == 0) {
      *ptr = 0;
      return cudaSuccess;
    } else {
      CUresult ret = cuMemAllocHost(ptr, size);
      if (ret == CUDA_SUCCESS) return cudaSuccess;
      assert(ret == CUDA_ERROR_OUT_OF_MEMORY);
      return cudaErrorMemoryAllocation;
    }
  }

  REALM_PUBLIC_API
  cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
  {
    get_gpu_or_die("cudaHostAlloc");

    // size=0 is allowed, but the runtime doesn't always do a good job of
    //  setting *ptr to nullptr, so do it ourselves
    if(size == 0) {
      *ptr = 0;
      return cudaSuccess;
    } else {
      CUresult ret = cuMemHostAlloc(ptr, size, flags);
      if (ret == CUDA_SUCCESS) return cudaSuccess;
      assert(ret == CUDA_ERROR_OUT_OF_MEMORY);
      return cudaErrorMemoryAllocation;
    }
  }

#if CUDART_VERSION >= 8000
  // Managed memory is only supported for cuda version >=8.0
  REALM_PUBLIC_API
  cudaError_t cudaMallocManaged(void **ptr, size_t size,
				unsigned flags /*= cudaMemAttachGlobal*/)
  {
    get_gpu_or_die("cudaMallocManaged");

    // size=0 is disallowed by the runtime api, and we need to return
    //  cudaErrorInvalidValue instead of calling into the driver and having it
    //  get mad
    if(size == 0) {
      // the cuda runtime actually clears this pointer as well
      *ptr = 0;
      return cudaErrorInvalidValue;
    } else {
      assert((flags == cudaMemAttachGlobal) || (flags == cudaMemAttachHost));
      CUresult ret = cuMemAllocManaged((CUdeviceptr*)ptr, size,
				       ((flags == cudaMemAttachGlobal) ?
					  CU_MEM_ATTACH_GLOBAL :
					  CU_MEM_ATTACH_HOST));
      if (ret == CUDA_SUCCESS) return cudaSuccess;
      assert(ret == CUDA_ERROR_OUT_OF_MEMORY);
      return cudaErrorMemoryAllocation;
    }
  }

  REALM_PUBLIC_API
  cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attr,
				       const void *ptr)
  {
    get_gpu_or_die("cudaPointerGetAttributes");

    unsigned mtype = 0;
    CUdeviceptr dptr = 0;
    void *hptr = 0;
    int ordinal = 0;
    // the docs say 'managed' is a bool, but the driver treats it as a
    //  32-bit integer
    int managed = 0;

    CUcontext ctx;
    {
      CUresult res = cuPointerGetAttribute(&ctx,
					   CU_POINTER_ATTRIBUTE_CONTEXT,
					   (CUdeviceptr)ptr);
      if(res == CUDA_ERROR_INVALID_VALUE) {
#if CUDART_VERSION >= 11000
	// starting with 11.0, cudart returns success for an unknown host pointer
	attr->type = cudaMemoryTypeUnregistered;
	attr->device = 0;
	attr->devicePointer = 0;
	attr->hostPointer = const_cast<void *>(ptr);
	return cudaSuccess;
#else
	return cudaErrorInvalidValue;
#endif
      }
      assert(res == CUDA_SUCCESS);
    }

    CHECK_CU( cuPointerGetAttribute(&mtype,
				    CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
				    (CUdeviceptr)ptr) );

    // requests for device and host pointers are allowed to fail if they're
    //  not accessible from device/host respectively
    {
      CUresult res = cuPointerGetAttribute(&dptr,
					   CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
					   (CUdeviceptr)ptr);
      if(res == CUDA_ERROR_INVALID_VALUE) {
	// we'll sanity check the pointers below
	dptr = 0;
      } else
	assert(res == CUDA_SUCCESS);
    }
    {
      CUresult res = cuPointerGetAttribute(&hptr,
					   CU_POINTER_ATTRIBUTE_HOST_POINTER,
					   (CUdeviceptr)ptr);
      if(res == CUDA_ERROR_INVALID_VALUE) {
	// we'll sanity check the pointers below
	hptr = 0;
      } else
	assert(res == CUDA_SUCCESS);
    }

#if CUDART_VERSION >= 9200
    CHECK_CU( cuPointerGetAttribute(&ordinal,
				    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
				    (CUdeviceptr)ptr) );
#else
    // have to get the device ordinal from the context
    CHECK_CU( cuCtxPushCurrent(ctx) );
    CHECK_CU( cuCtxGetDevice(&ordinal) );
    CHECK_CU( cuCtxPopCurrent(&ctx) );
#endif

    CHECK_CU( cuPointerGetAttribute(&managed,
				    CU_POINTER_ATTRIBUTE_IS_MANAGED,
				    (CUdeviceptr)ptr) );

    if(managed) {
#if CUDART_VERSION < 11000
      attr->memoryType = cudaMemoryTypeDevice;
      attr->isManaged = 1;
#endif
#if CUDART_VERSION >= 10000
      attr->type = cudaMemoryTypeManaged;
#endif
      // should have both host and device pointers
      assert((dptr != 0) && (hptr != 0));
    } else {
#if CUDART_VERSION < 11000
      attr->isManaged = 0;
#endif
      switch(mtype) {
      case CU_MEMORYTYPE_HOST: {
#if CUDART_VERSION < 11000
	attr->memoryType = cudaMemoryTypeHost;
#endif
#if CUDART_VERSION >= 10000
	attr->type = cudaMemoryTypeHost;
#endif
	assert(hptr != 0);
	break;
      }
      case CU_MEMORYTYPE_DEVICE: {
#if CUDART_VERSION < 11000
	attr->memoryType = cudaMemoryTypeDevice;
#endif
#if CUDART_VERSION >= 10000
	attr->type = cudaMemoryTypeDevice;
#endif
	assert(dptr != 0);
	break;
      }
      default:
	assert(0);
      }
    }

    attr->device = ordinal;
    attr->devicePointer = (void *)dptr;
    attr->hostPointer = hptr;

    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemPrefetchAsync(const void *dev_ptr,
				   size_t count,
				   int dst_device,
				   cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemPrefetchAsync");

    if(stream == 0)
      stream = p->gpu->get_null_task_stream()->get_stream();

    CHECK_CU( cuMemPrefetchAsync((CUdeviceptr)dev_ptr, count,
				 dst_device, stream) );
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemAdvise(const void *dev_ptr,
                            size_t count,
                            cudaMemoryAdvise advice,
                            int device)
  {
    /*GPUProcessor *p =*/ get_gpu_or_die("cudaMemAdvise");

    // NOTE: we assume the enums for cudaMeoryAdvise match the ones for
    //  CUmem_advise
    CHECK_CU( cuMemAdvise((CUdeviceptr)dev_ptr, count,
                          (CUmem_advise)advice, (CUdevice)device) );
    return cudaSuccess;
  }
#endif

  REALM_PUBLIC_API
  cudaError_t cudaFree(void *ptr)
  {
    /*GPUProcessor *p =*/ get_gpu_or_die("cudaFree");

    CUresult ret = cuMemFree((CUdeviceptr)ptr);
    if(ret == CUDA_SUCCESS) return cudaSuccess;
    assert(ret == CUDA_ERROR_INVALID_VALUE);
    return cudaErrorInvalidDevicePointer;
  }

  REALM_PUBLIC_API
  cudaError_t cudaFreeHost(void *ptr)
  {
    get_gpu_or_die("cudaFreeHost");
    CUresult ret = cuMemFreeHost(ptr);
    if (ret == CUDA_SUCCESS) return cudaSuccess;
    assert(ret == CUDA_ERROR_INVALID_VALUE);
    return cudaErrorInvalidHostPointer;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpy(void *dst, const void *src, 
			 size_t size, cudaMemcpyKind kind)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemcpy");
    p->gpu_memcpy(dst, src, size);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpyAsync(void *dst, const void *src,
			      size_t size, cudaMemcpyKind kind,
			      cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemcpyAsync");
    p->gpu_memcpy_async(dst, src, size, stream);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
			   size_t width, size_t height, cudaMemcpyKind kind)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemcpy2D");
    p->gpu_memcpy2d(dst, dpitch, src, spitch, width, height);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, 
				size_t spitch, size_t width, size_t height, 
				cudaMemcpyKind kind, cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemcpy2DAsync");
    p->gpu_memcpy2d_async(dst, dpitch, src, spitch, width, height, stream);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemset(void *dst, int value, size_t count)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemset");
    p->gpu_memset(dst, value, count);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemsetAsync(void *dst, int value, 
			      size_t count, cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemsetAsync");
    p->gpu_memset_async(dst, value, count, stream);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaDeviceSynchronize(void)
  {
    cudart_hijack_active = true;

    // instead of get_gpu_or_die, we can allow poorly written code to
    //  call cudaDeviceSynchronize from non-GPU tasks based on the
    //  '-cuda:nongpusync' command-line parameter
    //GPUProcessor *p = get_gpu_or_die("cudaDeviceSynchronize");
    GPUProcessor *p = GPUProcessor::get_current_gpu_proc();
    if(!p) {
      switch(cudart_hijack_nongpu_sync) {
      case 0: // ignore
	{
	  // don't complain, but there's nothing to synchronize against
	  //  either
	  return cudaSuccess;
	}

      case 1: // warn
	{
	  Backtrace bt;
	  bt.capture_backtrace();
	  bt.lookup_symbols();
	  log_cudart.warning() << "cudaDeviceSynchronize() called outside CUDA task at " << bt;
	  return cudaSuccess;
	}

      case 2: // fatal error
      default:
	{
	  log_cudart.fatal() << "cudaDeviceSynchronize() called outside CUDA task";
	  abort();
	}
      }
    }
    p->device_synchronize();
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpyToSymbol(const void *dst, const void *src,
				 size_t size, size_t offset,
				 cudaMemcpyKind kind)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemcpyToSymbol");
    p->gpu_memcpy_to_symbol(dst, src, size, offset);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpyToSymbolAsync(const void *dst, const void *src,
				      size_t size, size_t offset,
				      cudaMemcpyKind kind, cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemcpyToSymbolAsync");
    p->gpu_memcpy_to_symbol_async(dst, src, size, offset, stream);
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaMemcpyFromSymbol(void *dst, const void *src,
				   size_t size, size_t offset,
				   cudaMemcpyKind kind)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemcpyFromSymbol");
    p->gpu_memcpy_from_symbol(dst, src, size, offset);
    return cudaSuccess;
  }
      
  REALM_PUBLIC_API
  cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *src,
					size_t size, size_t offset,
					cudaMemcpyKind kind, cudaStream_t stream)
  {
    GPUProcessor *p = get_gpu_or_die("cudaMemcpyFromSymbolAsync");
    p->gpu_memcpy_from_symbol_async(dst, src, size, offset, stream);
    return cudaSuccess;
  }
      
  REALM_PUBLIC_API
  cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config)
  {
    /*GPUProcessor *p =*/ get_gpu_or_die("cudaDeviceSetSharedMemConfig");

    CUsharedconfig cfg = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
    if(config == cudaSharedMemBankSizeFourByte)
      cfg = CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
    if(config == cudaSharedMemBankSizeEightByte)
      cfg = CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE;
    CHECK_CU( cuCtxSetSharedMemConfig(cfg) );
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetDevice(int *device)
  {
    GPUProcessor *p = get_gpu_or_die("cudaGetDevice");
	
    // this wants the integer index, not the CUdevice
    *device = p->gpu->info->index;
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int index)
  {
    GPUProcessor *p = get_gpu_or_die("cudaGetDeviceProperties");
    const std::vector<GPUInfo*> &infos = p->gpu->module->gpu_info;
    for (const GPUInfo* info : infos) {
      if (info->index != index)
        continue;
      static_assert(std::is_trivially_copyable<cudaDeviceProp>::value, "cudaDeviceProp is no longer trivially copyable");
      memcpy(prop, info, sizeof(cudaDeviceProp));
      return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
  }
      
  REALM_PUBLIC_API
  cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int index)
  {
    // doesn't need a current device - the index is supplied
    CUdevice device;
    CHECK_CU( cuDeviceGet(&device, index) );
    // the runtime and device APIs appear to agree on attribute IDs, so just send it through
    //  and hope for the best
    CUdevice_attribute cu_attr = (CUdevice_attribute)attr;
    CHECK_CU( cuDeviceGetAttribute(value, cu_attr, device) );
    return cudaSuccess;
  } 

  REALM_PUBLIC_API
  cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, const void *func)
  {
    GPUProcessor *p = get_gpu_or_die("cudaFuncGetAttributes");

    CUfunction handle = p->gpu->lookup_function(func);

#define GET_FUNC_ATTR(member, name)					\
    do {								\
      int tmp;								\
      CHECK_CU( cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_##name, handle) ); \
      attr->member = tmp;						\
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
    GPUProcessor *p = get_gpu_or_die("cudaFuncSetAttribute");

    CUfunction handle = p->gpu->lookup_function(func);
    switch (attr)
    {
      case cudaFuncAttributeMaxDynamicSharedMemorySize:
        {
          CHECK_CU( cuFuncSetAttribute(handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value) );
          return cudaSuccess;
        }
      case cudaFuncAttributePreferredSharedMemoryCarveout:
        {
          CHECK_CU( cuFuncSetAttribute(handle, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, value) );
          return cudaSuccess;
        }
      default:
        break;
    }
    return cudaErrorInvalidValue;
  }
#endif

  REALM_PUBLIC_API
  cudaError_t cudaFuncSetCacheConfig(const void *func, cudaFuncCache cacheConfig)
  {
    GPUProcessor *p = get_gpu_or_die("cudaFuncSetCacheConfig");

    CUfunction handle = p->gpu->lookup_function(func);
    switch (cacheConfig)
    {
      case cudaFuncCachePreferNone:
        {
          CHECK_CU( cuFuncSetCacheConfig(handle, CU_FUNC_CACHE_PREFER_NONE) ); 
          return cudaSuccess;
        }
      case cudaFuncCachePreferShared:
        {
          CHECK_CU( cuFuncSetCacheConfig(handle, CU_FUNC_CACHE_PREFER_SHARED) ); 
          return cudaSuccess;
        }
      case cudaFuncCachePreferL1:
        {
          CHECK_CU( cuFuncSetCacheConfig(handle, CU_FUNC_CACHE_PREFER_L1) ); 
          return cudaSuccess;
        }
      case cudaFuncCachePreferEqual:
        {
          CHECK_CU( cuFuncSetCacheConfig(handle, CU_FUNC_CACHE_PREFER_EQUAL) ); 
          return cudaSuccess;
        }
      default:
        break;
    }
    return cudaErrorInvalidValue;
  }

  REALM_PUBLIC_API
  cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache config)
  {
    get_gpu_or_die("cudaDeviceSetCacheConfig");
    switch (config)
    {
      case cudaFuncCachePreferNone:
        {
          CHECK_CU( cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_NONE) ); 
          return cudaSuccess;
        }
      case cudaFuncCachePreferShared:
        {
          CHECK_CU( cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED) ); 
          return cudaSuccess;
        }
      case cudaFuncCachePreferL1:
        {
          CHECK_CU( cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1) ); 
          return cudaSuccess;
        }
      case cudaFuncCachePreferEqual:
        {
          CHECK_CU( cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_EQUAL) ); 
          return cudaSuccess;
        }
      default:
        break;
    }
    return cudaErrorInvalidValue;
  }

  REALM_PUBLIC_API
  cudaError_t cudaFuncSetSharedMemConfig(const void *func, cudaSharedMemConfig config)
  {
    GPUProcessor *p = get_gpu_or_die("cudaFuncSetSharedMemConfig");

    CUfunction handle = p->gpu->lookup_function(func);
    switch (config)
    {
      case cudaSharedMemBankSizeDefault:
        {
          CHECK_CU( cuFuncSetSharedMemConfig(handle, CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE) );
          return cudaSuccess;
        }
      case cudaSharedMemBankSizeFourByte:
        {
          CHECK_CU( cuFuncSetSharedMemConfig(handle, CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE) );
          return cudaSuccess;
        }
      case cudaSharedMemBankSizeEightByte:
        {
          CHECK_CU( cuFuncSetSharedMemConfig(handle, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE) );
          return cudaSuccess;
        }
      default:
        break;
    }
    return cudaErrorInvalidValue;
  }

#if CUDA_VERSION >= 11000
  REALM_PUBLIC_API
  cudaError_t cudaGetFuncBySymbol(cudaFunction_t *funcPtr, const void *symbolPtr)
  {
    GPUProcessor *p = get_gpu_or_die("cudaGetFuncBySymbol");
    CUfunction handle = p->gpu->lookup_function(symbolPtr);
    *funcPtr = handle;
    return cudaSuccess;
  }
#endif

  REALM_PUBLIC_API
  cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
							    const void *func,
							    int blockSize,
							    size_t dynamicSMemSize)
  {
    GPUProcessor *p = get_gpu_or_die("cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    CUfunction handle = p->gpu->lookup_function(func);
    CHECK_CU( cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, handle,
							  blockSize, dynamicSMemSize) );
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, 
								     const void *func, 
								     int blockSize, 
								     size_t dynamicSMemSize,
								     unsigned int flags)
  {
    GPUProcessor *p = get_gpu_or_die("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    CUfunction handle = p->gpu->lookup_function(func);
    CHECK_CU( cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, handle,
								   blockSize, dynamicSMemSize, flags) );
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetLastError(void)
  {
    get_gpu_or_die("cudaGetLastError");
    // For now we're not tracking this so if we were
    // going to die we already would have
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaPeekAtLastError(void)
  {
    get_gpu_or_die("cudaPeekAtLastError");
    // For now we're not tracking this so if we were
    // going to die we already would have
    return cudaSuccess;
  }

  REALM_PUBLIC_API
  const char* cudaGetErrorName(cudaError_t error)
  {
    const char *result = NULL;
    // It wasn't always the case the cuda runtime errors
    // and cuda driver errors had the same enum scheme
    // but it appears that they do now
    CHECK_CU( cuGetErrorName((CUresult)error, &result) );
    return result;
  }

  REALM_PUBLIC_API
  const char* cudaGetErrorString(cudaError_t error)
  {
    const char *result = NULL;
    // It wasn't always the case the cuda runtime errors
    // and cuda driver errors had the same enum scheme
    // but it appears that they do now
    CHECK_CU( cuGetErrorString((CUresult)error, &result) );
    return result;
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetDeviceCount(int *count)
  {
    // TODO: lie here and report just one device?
    CUresult res = cuDeviceGetCount(count);
    if(res == CUDA_SUCCESS)
      return cudaSuccess;
    else
      return cudaErrorInvalidValue;
  }

  REALM_PUBLIC_API
  cudaError_t cudaSetDevice(int device)
  {
    get_gpu_or_die("cudaSetDevice");
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
    get_gpu_or_die("cudaDeviceSetLimit");
    switch(limit) {
    case cudaLimitStackSize:
      return (cudaError_t)cuCtxSetLimit(CU_LIMIT_STACK_SIZE, value);

    case cudaLimitPrintfFifoSize:
      return (cudaError_t)cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, value);

    case cudaLimitMallocHeapSize:
      return (cudaError_t)cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, value);

    case cudaLimitDevRuntimeSyncDepth:
      return (cudaError_t)cuCtxSetLimit(CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, value);

    case cudaLimitDevRuntimePendingLaunchCount:
      return (cudaError_t)cuCtxSetLimit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, value);

#if CUDA_VERSION >= 10000
    case cudaLimitMaxL2FetchGranularity:
      return (cudaError_t)cuCtxSetLimit(CU_LIMIT_MAX_L2_FETCH_GRANULARITY, value);
#endif

#if CUDA_VERSION >= 11000
    case cudaLimitPersistingL2CacheSize:
      return (cudaError_t)cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, value);
#endif

    default:
      return cudaErrorInvalidValue;
    }
  }

  REALM_PUBLIC_API
  cudaError_t cudaDeviceGetLimit(size_t *value, cudaLimit limit)
  {
    get_gpu_or_die("cudaDeviceGetLimit");
    switch(limit) {
    case cudaLimitStackSize:
      return (cudaError_t)cuCtxGetLimit(value, CU_LIMIT_STACK_SIZE);

    case cudaLimitPrintfFifoSize:
      return (cudaError_t)cuCtxGetLimit(value, CU_LIMIT_PRINTF_FIFO_SIZE);

    case cudaLimitMallocHeapSize:
      return (cudaError_t)cuCtxGetLimit(value, CU_LIMIT_MALLOC_HEAP_SIZE);

    case cudaLimitDevRuntimeSyncDepth:
      return (cudaError_t)cuCtxGetLimit(value, CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH);

    case cudaLimitDevRuntimePendingLaunchCount:
      return (cudaError_t)cuCtxGetLimit(value, CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT);

#if CUDA_VERSION >= 10000
    case cudaLimitMaxL2FetchGranularity:
      return (cudaError_t)cuCtxGetLimit(value, CU_LIMIT_MAX_L2_FETCH_GRANULARITY);
#endif

#if CUDA_VERSION >= 11000
    case cudaLimitPersistingL2CacheSize:
      return (cudaError_t)cuCtxGetLimit(value, CU_LIMIT_PERSISTING_L2_CACHE_SIZE);
#endif

    default:
      return cudaErrorInvalidValue;
    }
  }

  REALM_PUBLIC_API
  cudaError_t cudaSetDeviceFlags(unsigned flags)
  {
    get_gpu_or_die("cudaSetDeviceFlags");

    // flags must be set before contexts are created, which is before any
    //  task can run in Realm, so we know it's too late
    return cudaErrorSetOnActiveProcess;
  }

  REALM_PUBLIC_API
  cudaError_t cudaGetDeviceFlags(unsigned *flags)
  {
    get_gpu_or_die("cudaGetDeviceFlags");

    // we're going to pass back the flags from the driver API, so assert
    //  the bits are in the same places
    assert((cudaDeviceScheduleAuto == CU_CTX_SCHED_AUTO) &&
           (cudaDeviceScheduleSpin == CU_CTX_SCHED_SPIN) &&
           (cudaDeviceScheduleYield == CU_CTX_SCHED_YIELD) &&
           (cudaDeviceScheduleBlockingSync == CU_CTX_SCHED_BLOCKING_SYNC) &&
           (cudaDeviceMapHost == CU_CTX_MAP_HOST) &&
           (cudaDeviceLmemResizeToMax == CU_CTX_LMEM_RESIZE_TO_MAX));
    return (cudaError_t)cuCtxGetFlags(flags);
  }

  REALM_PUBLIC_API
  cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                                const cudaChannelFormatDesc *desc,
                                cudaExtent extent, unsigned flags)
  {
    get_gpu_or_die("cudaMalloc3DArray");

    CUarray handle;
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

    CUresult ret = cuArray3DCreate(&handle, &d);
    if(ret == CUDA_SUCCESS) {
      *array = (struct cudaArray *)handle;
      return cudaSuccess;
    } else
      return (cudaError_t)ret;
  }

  REALM_PUBLIC_API
  cudaError_t cudaFreeArray(cudaArray_t array)
  {
    get_gpu_or_die("cudaFreeArray");

    return (cudaError_t)cuArrayDestroy((CUarray)array);
  }

  REALM_PUBLIC_API
  cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *object,
                                      const cudaResourceDesc *desc)
  {
    get_gpu_or_die("cudaCreateSurfaceObject");

    if(desc->resType != cudaResourceTypeArray)
      return cudaErrorInvalidValue;

    CUDA_RESOURCE_DESC d;
    memset(&d, 0, sizeof(d));
    d.resType = CU_RESOURCE_TYPE_ARRAY;
    d.res.array.hArray = (CUarray)(desc->res.array.array);
    d.flags = 0;

    CUsurfObject surf;
    CUresult ret = cuSurfObjectCreate(&surf, &d);
    if(ret == CUDA_SUCCESS) {
      *object = (cudaSurfaceObject_t)surf;
      return cudaSuccess;
    } else
      return (cudaError_t)ret;
  }

  REALM_PUBLIC_API
  cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t object)
  {
    get_gpu_or_die("cudaDestroySurfaceObject");

    return (cudaError_t)cuSurfObjectDestroy((CUsurfObject)object);
  }

}; // extern "C"
