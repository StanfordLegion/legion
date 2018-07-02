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

#include "realm/cuda/cudart_hijack.h"

#include "realm/cuda/cuda_module.h"
#include "realm/logging.h"

namespace Realm {
  namespace Cuda {

    extern Logger log_cudart;

    ////////////////////////////////////////////////////////////////////////
    //
    // CUDA Runtime API

    // these are all "C" functions
    extern "C" {
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
								      global != 0));
      }
      
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

      static GPUProcessor *get_gpu_or_die(const char *funcname)
      {
	// mark that the hijack code is active - this covers the calls below
	cudart_hijack_active = true;

	GPUProcessor *p = GPUProcessor::get_current_gpu_proc();
	if(!p) {
	  log_cudart.fatal() << funcname << "() called outside CUDA task";
	  assert(false);
	}
	return p;
      }

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

      cudaError_t cudaEventCreate(cudaEvent_t *event)
      {
	GPUProcessor *p = get_gpu_or_die("cudaEventCreate");
	p->event_create(event, cudaEventDefault);
	return cudaSuccess;
      }
	
      cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
					   unsigned int flags)
      {
	GPUProcessor *p = get_gpu_or_die("cudaEventCreateWithFlags");
	p->event_create(event, flags);
	return cudaSuccess;
      }

      cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0)
      {
	GPUProcessor *p = get_gpu_or_die("cudaEventRecord");
	p->event_record(event, stream);
	return cudaSuccess;
      }

      cudaError_t cudaEventSynchronize(cudaEvent_t event)
      {
	GPUProcessor *p = get_gpu_or_die("cudaEventSynchronize");
	p->event_synchronize(event);
	return cudaSuccess;
      }

      cudaError_t cudaEventDestroy(cudaEvent_t event)
      {
	GPUProcessor *p = get_gpu_or_die("cudaEventDestroy");
	p->event_destroy(event);
	return cudaSuccess;
      }

      cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
      {
	GPUProcessor *p = get_gpu_or_die("cudaEventElapsedTime");
	p->event_elapsed_time(ms, start, end);
	return cudaSuccess;
      }

      cudaError_t cudaStreamCreate(cudaStream_t *stream)
      {
	GPUProcessor *p = get_gpu_or_die("cudaStreamCreate");
        // For now we always return the stream for this task in case the user actually uses it
	// TODO: actually create sub-streams and connect them up
	*stream = p->gpu->get_current_task_stream()->get_stream();
	return cudaSuccess;
      }

      cudaError_t cudaStreamDestroy(cudaStream_t stream)
      {
	/*GPUProcessor *p =*/ get_gpu_or_die("cudaStreamDestroy");
        // Ignore this for now
        // TODO: Do the right thing if we are making substreams
	return cudaSuccess;
      }

      cudaError_t cudaStreamSynchronize(cudaStream_t stream)
      {
	GPUProcessor *p = get_gpu_or_die("cudaStreamSynchronize");
	p->stream_synchronize(stream);
	return cudaSuccess;
      }

      cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
				      cudaEvent_t event,
				      unsigned int flags)
      {
	/*GPUProcessor *p =*/ get_gpu_or_die("cudaStreamWaitEvent");
	// since we don't support user-level streams yet, this falls through to cudaWaitSynchronize
	return cudaEventSynchronize(event);
      }

      cudaError_t cudaConfigureCall(dim3 grid_dim,
				    dim3 block_dim,
				    size_t shared_memory,
				    cudaStream_t stream)
      {
	GPUProcessor *p = get_gpu_or_die("cudaConfigureCall");
	p->configure_call(grid_dim, block_dim, shared_memory, stream);
	return cudaSuccess;
      }

      cudaError_t cudaSetupArgument(const void *arg,
				    size_t size,
				    size_t offset)
      {
	GPUProcessor *p = get_gpu_or_die("cudaSetupArgument");
	p->setup_argument(arg, size, offset);
	return cudaSuccess;
      }

      cudaError_t cudaLaunch(const void *func)
      {
	GPUProcessor *p = get_gpu_or_die("cudaLaunch");
	p->launch(func);
	return cudaSuccess;
      }

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

      cudaError_t cudaMalloc(void **ptr, size_t size)
      {
	/*GPUProcessor *p =*/ get_gpu_or_die("cudaMalloc");

	CUresult ret = cuMemAlloc((CUdeviceptr *)ptr, size);
	if(ret == CUDA_SUCCESS) return cudaSuccess;
	assert(ret == CUDA_ERROR_OUT_OF_MEMORY);
	return cudaErrorMemoryAllocation;
      }

      cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
      {
        get_gpu_or_die("cudaHostAlloc");
        CUresult ret = cuMemHostAlloc(ptr, size, flags);
        if (ret == CUDA_SUCCESS) return cudaSuccess;
        assert(ret == CUDA_ERROR_OUT_OF_MEMORY);
        return cudaErrorMemoryAllocation;
      }

      cudaError_t cudaFree(void *ptr)
      {
	/*GPUProcessor *p =*/ get_gpu_or_die("cudaFree");

	CUresult ret = cuMemFree((CUdeviceptr)ptr);
	if(ret == CUDA_SUCCESS) return cudaSuccess;
	assert(ret == CUDA_ERROR_INVALID_VALUE);
	return cudaErrorInvalidDevicePointer;
      }

      cudaError_t cudaFreeHost(void *ptr)
      {
        get_gpu_or_die("cudaFreeHost");
        CUresult ret = cuMemFreeHost(ptr);
        if (ret == CUDA_SUCCESS) return cudaSuccess;
        assert(ret == CUDA_ERROR_INVALID_VALUE);
        return cudaErrorInvalidHostPointer;
      }

      cudaError_t cudaMemcpy(void *dst, const void *src, 
			     size_t size, cudaMemcpyKind kind)
      {
	GPUProcessor *p = get_gpu_or_die("cudaMemcpy");
	p->gpu_memcpy(dst, src, size, kind);
	return cudaSuccess;
      }

      cudaError_t cudaMemcpyAsync(void *dst, const void *src,
				  size_t size, cudaMemcpyKind kind,
				  cudaStream_t stream)
      {
	GPUProcessor *p = get_gpu_or_die("cudaMemcpyAsync");
	p->gpu_memcpy_async(dst, src, size, kind, stream);
	return cudaSuccess;
      }

      cudaError_t cudaMemset(void *dst, int value, size_t count)
      {
        GPUProcessor *p = get_gpu_or_die("cudaMemset");
        p->gpu_memset(dst, value, count);
        return cudaSuccess;
      }

      cudaError_t cudaMemsetAsync(void *dst, int value, 
                                  size_t count, cudaStream_t stream)
      {
        GPUProcessor *p = get_gpu_or_die("cudaMemsetAsync");
        p->gpu_memset_async(dst, value, count, stream);
        return cudaSuccess;
      }

      cudaError_t cudaDeviceSynchronize(void)
      {
	GPUProcessor *p = get_gpu_or_die("cudaDeviceSynchronize");
	p->device_synchronize();
	return cudaSuccess;
      }

      cudaError_t cudaMemcpyToSymbol(const void *dst, const void *src,
				     size_t size, size_t offset,
				     cudaMemcpyKind kind)
      {
	GPUProcessor *p = get_gpu_or_die("cudaMemcpyToSymbol");
	p->gpu_memcpy_to_symbol(dst, src, size, offset, kind);
	return cudaSuccess;
      }

      cudaError_t cudaMemcpyToSymbolAsync(const void *dst, const void *src,
					  size_t size, size_t offset,
					  cudaMemcpyKind kind, cudaStream_t stream)
      {
	GPUProcessor *p = get_gpu_or_die("cudaMemcpyToSymbolAsync");
	p->gpu_memcpy_to_symbol_async(dst, src, size, offset, kind, stream);
	return cudaSuccess;
      }

      cudaError_t cudaMemcpyFromSymbol(void *dst, const void *src,
				       size_t size, size_t offset,
				       cudaMemcpyKind kind)
      {
	GPUProcessor *p = get_gpu_or_die("cudaMemcpyFromSymbol");
	p->gpu_memcpy_from_symbol(dst, src, size, offset, kind);
	return cudaSuccess;
      }
      
      cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *src,
					    size_t size, size_t offset,
					    cudaMemcpyKind kind, cudaStream_t stream)
      {
	GPUProcessor *p = get_gpu_or_die("cudaMemcpyFromSymbolAsync");
	p->gpu_memcpy_from_symbol_async(dst, src, size, offset, kind, stream);
	return cudaSuccess;
      }
      
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
      
      const char* cudaGetErrorString(cudaError_t error)
      {
	// device and driver error strings don't match up...
	return "cudaGetErrorString() not supported";
      }

      cudaError_t cudaGetDevice(int *device)
      {
	GPUProcessor *p = get_gpu_or_die("cudaGetDevice");
	
	// this wants the integer index, not the CUdevice
	*device = p->gpu->info->index;
	return cudaSuccess;
      }

      cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int index)
      {
	// doesn't need a current device - the index is supplied
	CUdevice device;
	CHECK_CU( cuDeviceGet(&device, index) );
	CHECK_CU( cuDeviceGetName(prop->name, 255, device) );
	CHECK_CU( cuDeviceTotalMem(&(prop->totalGlobalMem), device) );
#define GET_DEVICE_PROP(member, name)   \
	do {				\
	  int tmp;							\
	  CHECK_CU( cuDeviceGetAttribute(&tmp, CU_DEVICE_ATTRIBUTE_##name, device) ); \
	  prop->member = tmp;						\
	} while(0)
	// SCREW TEXTURES AND SURFACES FOR NOW!
	GET_DEVICE_PROP(sharedMemPerBlock, MAX_SHARED_MEMORY_PER_BLOCK);
	GET_DEVICE_PROP(regsPerBlock, MAX_REGISTERS_PER_BLOCK);
	GET_DEVICE_PROP(warpSize, WARP_SIZE);
	GET_DEVICE_PROP(memPitch, MAX_PITCH);
	GET_DEVICE_PROP(maxThreadsPerBlock, MAX_THREADS_PER_BLOCK);
	GET_DEVICE_PROP(maxThreadsDim[0], MAX_BLOCK_DIM_X);
	GET_DEVICE_PROP(maxThreadsDim[1], MAX_BLOCK_DIM_Y);
	GET_DEVICE_PROP(maxThreadsDim[2], MAX_BLOCK_DIM_Z);
	GET_DEVICE_PROP(maxGridSize[0], MAX_GRID_DIM_X);
	GET_DEVICE_PROP(maxGridSize[1], MAX_GRID_DIM_Y);
	GET_DEVICE_PROP(maxGridSize[2], MAX_GRID_DIM_Z);
	GET_DEVICE_PROP(clockRate, CLOCK_RATE);
	GET_DEVICE_PROP(totalConstMem, TOTAL_CONSTANT_MEMORY);
	GET_DEVICE_PROP(major, COMPUTE_CAPABILITY_MAJOR);
	GET_DEVICE_PROP(minor, COMPUTE_CAPABILITY_MINOR);
	GET_DEVICE_PROP(deviceOverlap, GPU_OVERLAP);
	GET_DEVICE_PROP(multiProcessorCount, MULTIPROCESSOR_COUNT);
	GET_DEVICE_PROP(kernelExecTimeoutEnabled, KERNEL_EXEC_TIMEOUT);
	GET_DEVICE_PROP(integrated, INTEGRATED);
	GET_DEVICE_PROP(canMapHostMemory, CAN_MAP_HOST_MEMORY);
	GET_DEVICE_PROP(computeMode, COMPUTE_MODE);
	GET_DEVICE_PROP(concurrentKernels, CONCURRENT_KERNELS);
	GET_DEVICE_PROP(ECCEnabled, ECC_ENABLED);
	GET_DEVICE_PROP(pciBusID, PCI_BUS_ID);
	GET_DEVICE_PROP(pciDeviceID, PCI_DEVICE_ID);
	GET_DEVICE_PROP(pciDomainID, PCI_DOMAIN_ID);
	GET_DEVICE_PROP(tccDriver, TCC_DRIVER);
	GET_DEVICE_PROP(asyncEngineCount, ASYNC_ENGINE_COUNT);
	GET_DEVICE_PROP(unifiedAddressing, UNIFIED_ADDRESSING);
	GET_DEVICE_PROP(memoryClockRate, MEMORY_CLOCK_RATE);
	GET_DEVICE_PROP(memoryBusWidth, GLOBAL_MEMORY_BUS_WIDTH);
	GET_DEVICE_PROP(l2CacheSize, L2_CACHE_SIZE);
	GET_DEVICE_PROP(maxThreadsPerMultiProcessor, MAX_THREADS_PER_MULTIPROCESSOR);
	GET_DEVICE_PROP(streamPrioritiesSupported, STREAM_PRIORITIES_SUPPORTED);
	GET_DEVICE_PROP(globalL1CacheSupported, GLOBAL_L1_CACHE_SUPPORTED);
	GET_DEVICE_PROP(localL1CacheSupported, LOCAL_L1_CACHE_SUPPORTED);
	GET_DEVICE_PROP(sharedMemPerMultiprocessor, MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
	GET_DEVICE_PROP(regsPerMultiprocessor, MAX_REGISTERS_PER_MULTIPROCESSOR);
	GET_DEVICE_PROP(managedMemory, MANAGED_MEMORY);
	GET_DEVICE_PROP(isMultiGpuBoard, MULTI_GPU_BOARD);
	GET_DEVICE_PROP(multiGpuBoardGroupID, MULTI_GPU_BOARD_GROUP_ID);
#undef GET_DEVICE_PROP
	return cudaSuccess;
      }
      
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

      cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, const void *func)
      {
	GPUProcessor *p = get_gpu_or_die("cudaFuncGetAttributes");

	CUfunction handle = p->gpu->lookup_function(func);

#define GET_FUNC_ATTR(member, name)   \
	do {			\
	  int tmp;							\
	  CHECK_CU( cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_##name, handle) ); \
	  attr->member = tmp;						\
	} while(0)

	GET_FUNC_ATTR(binaryVersion, BINARY_VERSION);
	GET_FUNC_ATTR(cacheModeCA, CACHE_MODE_CA);
	GET_FUNC_ATTR(constSizeBytes, CONST_SIZE_BYTES);
	GET_FUNC_ATTR(localSizeBytes, LOCAL_SIZE_BYTES);
	GET_FUNC_ATTR(maxThreadsPerBlock, MAX_THREADS_PER_BLOCK);
	GET_FUNC_ATTR(numRegs, NUM_REGS);
	GET_FUNC_ATTR(ptxVersion, PTX_VERSION);
	GET_FUNC_ATTR(sharedSizeBytes, SHARED_SIZE_BYTES);
#undef GET_FUNC_ATTR
	return cudaSuccess;
      }

    }; // extern "C"
  }; // namespace Cuda
}; // namespace Realm


