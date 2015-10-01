// Our implementation of the CUDA runtime API for Legion
// so we can intercept all of these calls

// All these extern C methods are for internal implementations
// of functions of the cuda runtime API that nvcc assumes
// exists and can be used for code generation. They are all
// pretty simple to map to the driver API.

// so that we get types and stuff right
#include <cuda_runtime.h>

#include "realm/cuda/cuda_module.h"

namespace Realm {
  namespace Cuda {

    // these are all "C" functions
    extern "C" {
      void** __cudaRegisterFatBinary(void *fat_bin)
      {
	return GPUProcessor::register_fat_binary(fat_bin);
      }

      // this is not really a part of CUDA runtime API but used by the regent compiler
      void** __cudaRegisterCudaBinary(void *cubin, size_t cubinSize)
      {
	return GPUProcessor::register_cuda_binary(cubin,
						  cubinSize);
      }

      void __cudaUnregisterFatBinary(void **fat_bin)
      {
	GPUProcessor::unregister_fat_binary(fat_bin);
      }

      void __cudaRegisterVar(void **fat_bin,
			     char *host_var,
			     char *device_addr,
			     const char *device_name,
			     int ext, int size, int constant, int global)
      {
	GPUProcessor::register_var(fat_bin, host_var, device_addr,
				   device_name, ext, size, 
				   constant, global);
      }
      
      void __cudaRegisterFunction(void **fat_bin,
				  const char *host_fun,
				  char *device_fun,
				  const char *device_name,
				  int thread_limit,
				  uint3 *tid, uint3 *bid,
				  dim3 *bDim, dim3 *gDim,
				  int *wSize)
      {
	GPUProcessor::register_function(fat_bin, host_fun,
					device_fun, device_name,
					thread_limit, tid, bid,
					bDim, gDim, wSize);
      }
      
      char __cudaInitModule(void **fat_bin)
      {
	return GPUProcessor::init_module(fat_bin);
      }

      // All the following methods are cuda runtime API calls that we 
      // intercept and then either execute using the driver API or 
      // modify in ways that are important to Legion.

      cudaError_t cudaStreamCreate(cudaStream_t *stream)
      {
	return GPUProcessor::stream_create(stream);
      }

      cudaError_t cudaStreamDestroy(cudaStream_t stream)
      {
	return GPUProcessor::stream_destroy(stream);
      }

      cudaError_t cudaStreamSynchronize(cudaStream_t stream)
      {
	return GPUProcessor::stream_synchronize(stream);
      }

      cudaError_t cudaConfigureCall(dim3 grid_dim,
				    dim3 block_dim,
				    size_t shared_memory,
				    cudaStream_t stream)
      {
	return GPUProcessor::configure_call(grid_dim, block_dim,
					    shared_memory, stream);
      }

      cudaError_t cudaSetupArgument(const void *arg,
				    size_t size,
				    size_t offset)
      {
	return GPUProcessor::setup_argument(arg, size, offset);
      }

      cudaError_t cudaLaunch(const void *func)
      {
	return GPUProcessor::launch(func);
      }

      cudaError_t cudaMalloc(void **ptr, size_t size)
      {
	return GPUProcessor::gpu_malloc(ptr, size);
      }

      cudaError_t cudaFree(void *ptr)
      {
	return GPUProcessor::gpu_free(ptr);
      }

      cudaError_t cudaMemcpy(void *dst, const void *src, 
			     size_t size, cudaMemcpyKind kind)
      {
	return GPUProcessor::gpu_memcpy(dst, src, size, kind);
      }

      cudaError_t cudaMemcpyAsync(void *dst, const void *src,
				  size_t size, cudaMemcpyKind kind,
				  cudaStream_t stream)
      {
	return GPUProcessor::gpu_memcpy_async(dst, src, size, kind, stream);
      }

      cudaError_t cudaDeviceSynchronize(void)
      {
	return GPUProcessor::device_synchronize();
      }

      cudaError_t cudaMemcpyToSymbol(const void *dst, const void *src,
				     size_t size, size_t offset,
				     cudaMemcpyKind kind)
      {
	return GPUProcessor::gpu_memcpy_to_symbol(dst, src, size, 
						  offset, kind, true/*sync*/);
      }

      cudaError_t cudaMemcpyToSymbolAsync(const void *dst, const void *src,
					  size_t size, size_t offset,
					  cudaMemcpyKind kind, cudaStream_t stream)
      {
	return GPUProcessor::gpu_memcpy_to_symbol(dst, src, size,
						  offset, kind, false/*sync*/);
      }

      cudaError_t cudaMemcpyFromSymbol(void *dst, const void *src,
				       size_t size, size_t offset,
				       cudaMemcpyKind kind)
      {
	return GPUProcessor::gpu_memcpy_from_symbol(dst, src, size,
						    offset, kind, true/*sync*/);
      }
      
      cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *src,
					    size_t size, size_t offset,
					    cudaMemcpyKind kind, cudaStream_t stream)
      {
	return GPUProcessor::gpu_memcpy_from_symbol(dst, src, size,
						    offset, kind, false/*sync*/);
      }
      
      cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config)
      {
	return GPUProcessor::set_shared_memory_config(config);
      }
      
      const char* cudaGetErrorString(cudaError_t error)
      {
	return GPUProcessor::get_error_string(error);
      }

      cudaError_t cudaGetDevice(int *device)
      {
	return GPUProcessor::get_device(device);
      }

      cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
      {
	return GPUProcessor::get_device_properties(prop, device);
      }

      cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, const void *func)
      {
	return GPUProcessor::get_func_attributes(attr, func);
      }

    }; // extern "C"
  }; // namespace Cuda
}; // namespace Realm


