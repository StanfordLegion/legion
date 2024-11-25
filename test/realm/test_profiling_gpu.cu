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