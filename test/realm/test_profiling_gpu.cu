#include "realm.h"
#ifdef REALM_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

__global__ void spin_kernel(uint64_t t_ns)
{
  uint64_t start = 0, current = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(current));
  } while((current - start) < t_ns);
}

void launch_spin_kernel(uint64_t t_ns, CUstream stream)
{
  void *args[] = {&t_ns};
  cudaError_t err = cudaLaunchKernel(reinterpret_cast<void *>(spin_kernel), dim3(1),
                                     dim3(1), args, 0, static_cast<cudaStream_t>(stream));
}
#endif

#ifdef REALM_USE_HIP
#include <hip/hip_runtime.h>

__global__ void spin_kernel(uint64_t t_ns)
{
  uint64_t start = clock64();
  uint64_t current = start;

  const uint64_t clock_frequency = 1000000; // MHz -> ns
  uint64_t target_cycles = (t_ns * clock_frequency) / 1000000000;

  while ((current - start) < target_cycles) {
    current = clock64();
  }
}

void launch_spin_kernel(uint64_t t_ns, hipStream_t stream)
{
  void *args[] = {&t_ns};
  hipError_t err = hipLaunchKernel(reinterpret_cast<void*>(spin_kernel), dim3(1), 
                                   dim3(1), args, 0, stream);
  if (err != hipSuccess) {
    printf("Error launching spin kernel: %s\n", hipGetErrorString(err));
  }
}
#endif