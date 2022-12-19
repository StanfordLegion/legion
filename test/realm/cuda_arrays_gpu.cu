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
__global__ void smooth(Realm::Rect<N,T> extent,
		       int nx, int ny, int nz,
		       float alpha,
                       cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;

  if((x >= nx) || (y >= ny) || (z >= nz)) return;

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
  size_t tx = extent.hi.x - extent.lo.x + 1;
  size_t bx = std::min<size_t>(1024, tx);
  size_t gx = 1 + ((tx - 1) / bx);
  dim3 gridDim(gx, 1, 1);
  dim3 blockDim(bx, 1, 1);

  smooth<<<gridDim, blockDim>>>(extent, tx, 1, 1, alpha, surf_in, surf_out);
}

void smooth_kernel(Realm::Rect<2> extent, float alpha,
                   cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out)
{
  size_t tx = extent.hi.x - extent.lo.x + 1;
  size_t bx = std::min<size_t>(1024, tx);
  size_t gx = 1 + ((tx - 1) / bx);
  size_t ty = extent.hi.y - extent.lo.y + 1;
  size_t by = std::min<size_t>(1024 / bx, ty);
  size_t gy = 1 + ((ty - 1) / by);
  dim3 gridDim(gx, gy, 1);
  dim3 blockDim(bx, by, 1);

  smooth<<<gridDim, blockDim>>>(extent, tx, ty, 1, alpha, surf_in, surf_out);
}

void smooth_kernel(Realm::Rect<3> extent, float alpha,
                   cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out)
{
  size_t tx = extent.hi.x - extent.lo.x + 1;
  size_t bx = std::min<size_t>(1024, tx);
  size_t gx = 1 + ((tx - 1) / bx);
  size_t ty = extent.hi.y - extent.lo.y + 1;
  size_t by = std::min<size_t>(1024 / bx, ty);
  size_t gy = 1 + ((ty - 1) / by);
  size_t tz = extent.hi.z - extent.lo.z + 1;
  size_t bz = std::min<size_t>(1024 / (bx * by), tz);
  size_t gz = 1 + ((tz - 1) / bz);
  dim3 gridDim(gx, gy, gz);
  dim3 blockDim(bx, by, bz);

  smooth<<<gridDim, blockDim>>>(extent, tx, ty, tz, alpha, surf_in, surf_out);
}

