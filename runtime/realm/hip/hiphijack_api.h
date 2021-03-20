#ifndef HIPHIJACK_API_H
#define HIPHIJACK_API_H

#ifdef __HIP_PLATFORM_NVCC__
#include "cuda_runtime.h"
extern "C" {
cudaStream_t hipGetTaskStream();
}
#else
#include "hip/hip_runtime.h"
extern "C" {
hipStream_t hipGetTaskStream();
}
#endif

#endif // HIPHIJACK_API_H
