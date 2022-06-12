#include <stdio.h>
#include <assert.h>

#include "realm_defines.h"

#ifdef REALM_USE_CUDA
#include <cuda.h>
//include <cuda_runtime.h>
#endif

#ifdef REALM_USE_HIP
#include "hip_cuda_compat/hip_cuda.h"
#include "realm/hip/hiphijack_api.h"
#endif

extern "C" {
  double gpu_seqwr_test(void *buffer, size_t reps, size_t elements);
  double gpu_seqrd_test(void *buffer, size_t reps, size_t elements);
  double gpu_rndwr_test(void *buffer, size_t reps, size_t elements);
  double gpu_rndrd_test(void *buffer, size_t reps, size_t elements);
  double gpu_latency_test(void *buffer, size_t reps, size_t elements);
}

__global__ void gpu_seqwr_kernel(int *buffer, size_t reps, size_t elements)
{
  for(size_t j = 0; j < reps; j++) {
    size_t ofs = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;
    while(ofs < elements) {
      buffer[ofs] = 0;
      ofs += step;
    }
  }
}

__global__ void gpu_seqrd_kernel(int *buffer, size_t reps, size_t elements)
{
  int errors = 0;
  for(size_t j = 0; j < reps; j++) {
    size_t ofs = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;
    while(ofs < elements) {
      // manually unroll loop to get multiple loads in flight per thread
      int val1 = buffer[ofs];
      ofs += step;
      int val2 = (ofs < elements) ? buffer[ofs] : 0;
      ofs += step;
      int val3 = (ofs < elements) ? buffer[ofs] : 0;
      ofs += step;
      int val4 = (ofs < elements) ? buffer[ofs] : 0;
      ofs += step;
      // now check result of all the reads
      if(val1 != 0) errors++;
      if(val2 != 0) errors++;
      if(val3 != 0) errors++;
      if(val4 != 0) errors++;
    }
  }
  if(errors > 0)
    buffer[0] = errors;
}

__global__ void gpu_rndwr_kernel(int *buffer, size_t reps, size_t steps, size_t elements)
{
  // we don't want completely random writes here, since the performance would be awful
  // instead, let each warp move around randomly, but keep the warp coalesced on 128B-aligned
  //  accesses
  for(size_t j = 0; j < reps; j++) {
    // starting point is naturally aligned
    size_t p = blockIdx.x * blockDim.x + threadIdx.x;
    // if we start outside the block, sit this out (just to keep small runs from crashing)
    if(p >= elements) break;

    // quadratic stepping via "acceleration" and "velocity"
    size_t a = 548191;
    size_t v = 24819 + (p >> 5);  // velocity has to be different for each warp

    for(size_t i = 0; i < steps; i++) {
      size_t prev = p;
      // delta is multiplied by 32 elements so warp stays converged (velocity is the
      //  same for all threads in the warp)
      p = (p + (v << 5)) % elements;
      v = (v + a) % elements;
      buffer[prev] = p;
    }
  }
}

__global__ void gpu_rndrd_kernel(int *buffer, size_t reps, size_t steps, size_t elements)
{
  // we don't want completely random writes here, since the performance would be awful
  // instead, let each warp move around randomly, but keep the warp coalesced on 128B-aligned
  //  accesses
  int errors = 0;
  for(size_t j = 0; j < reps; j++) {
    // starting point is naturally aligned
    size_t p = blockIdx.x * blockDim.x + threadIdx.x;
    // if we start outside the block, sit this out (just to keep small runs from crashing)
    if(p >= elements) break;

    // quadratic stepping via "acceleration" and "velocity"
    size_t a = 548191;
    size_t v = 24819 + (p >> 5);  // velocity has to be different for each warp

    for(size_t i = 0; i < steps; i += 4) {
      // delta is multiplied by 32 elements so warp stays converged (velocity is the
      //  same for all threads in the warp)
      // manually unroll loop to get multiple loads in flight per thread
      size_t p0 = p;
      p = (p + (v << 5)) % elements;
      v = (v + a) % elements;
      size_t p1 = p;
      p = (p + (v << 5)) % elements;
      v = (v + a) % elements;
      size_t p2 = p;
      p = (p + (v << 5)) % elements;
      v = (v + a) % elements;
      size_t p3 = p;
      p = (p + (v << 5)) % elements;
      v = (v + a) % elements;

      int v0 = buffer[p0];
      int v1 = buffer[p1];
      int v2 = buffer[p2];
      int v3 = buffer[p3];

      if(v0 != p1) errors++;
      if(v1 != p2) errors++;
      if(v2 != p3) errors++;
      if(v3 != p) errors++;
    }
  }
  if((errors > 0) && (reps > elements))
    buffer[0] = errors;
}

__global__ void gpu_latency_setup_kernel(int *buffer, size_t delta, size_t elements)
{
  size_t ofs = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;
  while(ofs < elements) {
    size_t tgt = ofs + delta;
    while(tgt >= elements)
      tgt -= elements;
    buffer[ofs] = tgt;
    ofs += step;
  }
}

__global__ void gpu_latency_kernel(int *buffer, size_t reps, size_t steps, size_t elements)
{
  int errors = 0;
  // this is done with just a single thread
  for(size_t j = 0; j < reps; j++) {
    int p = j & 31;

    for(size_t i = 0; i < steps; i++) {
      int next = buffer[p];

      if((next >= 0) && (next < elements)) {
	p = next;
      } else {
	printf("%d -> %d\n", p, next);
	p = 0;
	errors++;
      }
    }
  }
  if((errors > 0) && (reps > elements))
    buffer[0] = errors;
}	  

static void get_launch_params(int *grid_size, int *block_size)
{
 // want to fill the GPU precisely, so figure out how many threads we can fit
  //  (our register count should be low enough to not be limited by RF size)
  int device;
  struct cudaDeviceProp props;
  cudaError_t ret;

  ret = cudaGetDevice(&device);
  assert(ret == cudaSuccess);

  ret = cudaGetDeviceProperties(&props, device);
  assert(ret == cudaSuccess);

  // seems like this should be queryable?
  int ctas_per_sm = 8;
  int threads_per_sm = 512;
  int threads_per_cta = threads_per_sm / ctas_per_sm;
  int total_ctas = props.multiProcessorCount * ctas_per_sm;

  *grid_size = total_ctas;
  *block_size = threads_per_cta;
}

double gpu_seqwr_test(void *buffer, size_t reps, size_t elements)
{
  int grid_size, block_size;
  get_launch_params(&grid_size, &block_size);

  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start, 0);
  gpu_seqwr_kernel<<< grid_size, block_size
#ifdef REALM_USE_HIP
                      , 0, hipGetTaskStream()
#endif
                  >>>((int *)buffer, reps, elements);
  cudaEventRecord(t_end, 0);
  
  cudaError_t ret = cudaEventSynchronize(t_end);
  assert(ret == cudaSuccess);

  float elapsed;
  cudaEventElapsedTime(&elapsed, t_start, t_end);

  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);

  // BW units are GB/s (a.k.a. B/ns) - elapsed is in ms
  double seqwr_bw = 1e-6 * reps * elements * sizeof(int) / elapsed;
  return seqwr_bw;
}

double gpu_seqrd_test(void *buffer, size_t reps, size_t elements)
{
  int grid_size, block_size;
  get_launch_params(&grid_size, &block_size);

  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start, 0);
  gpu_seqrd_kernel<<< grid_size, block_size
#ifdef REALM_USE_HIP
                      , 0, hipGetTaskStream()
#endif
                  >>>((int *)buffer, reps, elements);
  cudaEventRecord(t_end, 0);
  
  cudaError_t ret = cudaEventSynchronize(t_end);
  assert(ret == cudaSuccess);

  float elapsed;
  cudaEventElapsedTime(&elapsed, t_start, t_end);

  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);

  // BW units are GB/s (a.k.a. B/ns) - elapsed is in ms
  double seqrd_bw = 1e-6 * reps * elements * sizeof(int) / elapsed;
  return seqrd_bw;
}

double gpu_rndwr_test(void *buffer, size_t reps, size_t elements)
{
  int grid_size, block_size;
  get_launch_params(&grid_size, &block_size);
  int total_threads = grid_size * block_size;

  size_t steps = 64;

  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start, 0);
  gpu_rndwr_kernel<<< grid_size, block_size
#ifdef REALM_USE_HIP
                      , 0, hipGetTaskStream()
#endif
                  >>>((int *)buffer, reps, steps, elements);
  cudaEventRecord(t_end, 0);
  
  cudaError_t ret = cudaEventSynchronize(t_end);
  assert(ret == cudaSuccess);

  float elapsed;
  cudaEventElapsedTime(&elapsed, t_start, t_end);

  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);

  // BW units are GB/s (a.k.a. B/ns) - elapsed is in ms
  double rndwr_bw = 1e-6 * reps * total_threads * steps * sizeof(int) / elapsed;
  return rndwr_bw;
}

double gpu_rndrd_test(void *buffer, size_t reps, size_t elements)
{
  int grid_size, block_size;
  get_launch_params(&grid_size, &block_size);
  int total_threads = grid_size * block_size;

  size_t steps = 64; // must be a multiple of 4 due to unrolling in kernel

  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start, 0);
  gpu_rndrd_kernel<<< grid_size, block_size
#ifdef REALM_USE_HIP
                      , 0, hipGetTaskStream()
#endif
                  >>>((int *)buffer, reps, steps, elements);
  cudaEventRecord(t_end, 0);
  
  cudaError_t ret = cudaEventSynchronize(t_end);
  assert(ret == cudaSuccess);

  float elapsed;
  cudaEventElapsedTime(&elapsed, t_start, t_end);

  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);

  // BW units are GB/s (a.k.a. B/ns) - elapsed is in ms
  double rndrd_bw = 1e-6 * reps * total_threads * steps * sizeof(int) / elapsed;
  return rndrd_bw;
}

double gpu_latency_test(void *buffer, size_t reps, size_t elements)
{
  int grid_size, block_size;
  get_launch_params(&grid_size, &block_size);

  size_t steps = 1024;

  // initialize the data with something that makes large jumps through memory
  // for now, assume that no prefetcher will take notice of the stride
  // try to pick something that won't get too close to a multiple of the element size
  //  in 'steps' tries
  size_t delta = (((steps >> 2) - 3) / (steps + 1.0)) * elements;
  if(delta == 0) delta = 1;
  gpu_latency_setup_kernel<<< grid_size, block_size
#ifdef REALM_USE_HIP
                          , 0, hipGetTaskStream()
#endif
                          >>>((int *)buffer, delta, elements);

  cudaEvent_t t_start, t_end;
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start, 0);
  gpu_latency_kernel<<< 1, 1
#ifdef REALM_USE_HIP
                    , 0, hipGetTaskStream()
#endif
                    >>>((int *)buffer, reps, steps, elements);
  cudaEventRecord(t_end, 0);
  
  cudaError_t ret = cudaEventSynchronize(t_end);
  assert(ret == cudaSuccess);

  float elapsed;
  cudaEventElapsedTime(&elapsed, t_start, t_end);

  cudaEventDestroy(t_start);
  cudaEventDestroy(t_end);

  // latency units are in ns - elapsed is in ms
  double latency = (elapsed * 1e6) / (reps * steps);
  return latency;
}
