#include "cuda_arrays.h"

template <int N>
__device__ float read_surface(cudaSurfaceObject_t s, int x, int y, int z);

template <>
__device__ float read_surface<1>(cudaSurfaceObject_t s, int x, int y, int z)
{
  return surf1Dread<float>(s, x * sizeof(float), cudaBoundaryModeClamp);
}

template <>
__device__ float read_surface<2>(cudaSurfaceObject_t s, int x, int y, int z)
{
  return surf2Dread<float>(s, x * sizeof(float), y, cudaBoundaryModeClamp);
}

template <>
__device__ float read_surface<3>(cudaSurfaceObject_t s, int x, int y, int z)
{
  return surf3Dread<float>(s, x * sizeof(float), y, z, cudaBoundaryModeClamp);
}

template <int N>
__device__ void write_surface(float f, cudaSurfaceObject_t s,
                              int x, int y, int z);

template <>
__device__ void write_surface<1>(float f, cudaSurfaceObject_t s,
                                 int x, int y, int z)
{
  surf1Dwrite<float>(f, s, x * sizeof(float));
}

template <>
__device__ void write_surface<2>(float f, cudaSurfaceObject_t s,
                                 int x, int y, int z)
{
  surf2Dwrite<float>(f, s, x * sizeof(float), y);
}

template <>
__device__ void write_surface<3>(float f, cudaSurfaceObject_t s,
                                 int x, int y, int z)
{
  surf3Dwrite<float>(f, s, x * sizeof(float), y, z);
}

template <int N, typename T>
__global__ void smooth(Realm::Rect<N,T> extent, float alpha,
                       cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;

  // 6-point stencil
  float f = (1 - alpha) * read_surface<N>(surf_in, x, y, z);

  f += alpha/6.0f * read_surface<N>(surf_in, x-1, y, z);
  f += alpha/6.0f * read_surface<N>(surf_in, x+1, y, z);
  f += alpha/6.0f * read_surface<N>(surf_in, x, y-1, z);
  f += alpha/6.0f * read_surface<N>(surf_in, x, y+1, z);
  f += alpha/6.0f * read_surface<N>(surf_in, x, y, z-1);
  f += alpha/6.0f * read_surface<N>(surf_in, x, y, z+1);

  write_surface<N>(f, surf_out, x, y, z);
}

void smooth_kernel(Realm::Rect<1> extent, float alpha,
                   cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out)
{
  dim3 gridDim(1, 1, 1);
  dim3 blockDim(extent.hi.x - extent.lo.x + 1, 1, 1);

  smooth<<<gridDim, blockDim>>>(extent, alpha, surf_in, surf_out);
}

void smooth_kernel(Realm::Rect<2> extent, float alpha,
                   cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out)
{
  dim3 gridDim(1, 1, 1);
  dim3 blockDim(extent.hi.x - extent.lo.x + 1,
                extent.hi.y - extent.lo.y + 1, 1);

  smooth<<<gridDim, blockDim>>>(extent, alpha, surf_in, surf_out);
}

void smooth_kernel(Realm::Rect<3> extent, float alpha,
                   cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out)
{
  dim3 gridDim(1, 1, 1);
  dim3 blockDim(extent.hi.x - extent.lo.x + 1,
                extent.hi.y - extent.lo.y + 1,
                extent.hi.z - extent.lo.z + 1);

  smooth<<<gridDim, blockDim>>>(extent, alpha, surf_in, surf_out);
}

