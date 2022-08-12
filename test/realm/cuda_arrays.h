#ifndef CUDA_ARRAYS_H
#define CUDA_ARRAYS_H

#include "realm.h"

#include <surface_types.h>

#define CHECK_CUDART(cmd) do { \
  cudaError_t ret = (cmd); \
  if(ret != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s)\n", #cmd, ret, cudaGetErrorString(ret)); \
    assert(0); \
    exit(1); \
  } \
} while(0)

void smooth_kernel(Realm::Rect<1> extent, float alpha,
                   cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out);
void smooth_kernel(Realm::Rect<2> extent, float alpha,
                   cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out);
void smooth_kernel(Realm::Rect<3> extent, float alpha,
                   cudaSurfaceObject_t surf_in, cudaSurfaceObject_t surf_out);

#endif
