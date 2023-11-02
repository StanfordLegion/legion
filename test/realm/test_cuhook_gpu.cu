/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm.h"
#include <cuda.h>

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

__global__ void empty_kernel() {}

void gpu_kernel_wrapper(cudaStream_t stream)
{
  float *d_A, *d_B, *d_C, *h_A, *h_B, *h_C;
  int N = 100;
  size_t size_matrix = sizeof(float) * N * N;
  h_A = (float *)malloc(size_matrix);
  h_B = (float *)malloc(size_matrix);
  h_C = (float *)malloc(size_matrix);
  for(int i = 0; i < N * N; i++) {
    h_A[i] = 1.0;
    h_B[i] = 1.0;
    h_C[i] = 1.0;
  }
  cudaMalloc((void **)&d_A, size_matrix);
  cudaMalloc((void **)&d_B, size_matrix);
  cudaMalloc((void **)&d_C, size_matrix);
  cudaMemcpyAsync(d_A, h_A, size_matrix, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_B, h_B, size_matrix, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_C, h_C, size_matrix, cudaMemcpyHostToDevice, stream);
#ifdef USE_CUBLAS
  float alpha = 1.0;
  float beta = 1.0;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta,
              d_C, N);
#else
  empty_kernel<<<1, 1, 0, stream>>>();
#endif
  cudaMemcpyAsync(h_A, d_A, size_matrix, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_B, d_B, size_matrix, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_C, d_C, size_matrix, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
#ifdef USE_CUBLAS
  cublasDestroy(handle);
  for(int i = 0; i < N * N; i++) {
    assert(h_C[i] == N + 1.0);
  }
#endif
  free(h_A);
  free(h_B);
  free(h_C);

  cudaEvent_t e1, e2;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  empty_kernel<<<1, 1, 0, stream>>>();
  empty_kernel<<<1, 1, 0, stream>>>();
  cudaEventRecord(e1, stream);
  cudaEventRecord(e2, stream);
  cudaEventSynchronize(e2);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);

  // since stream is a realm stream, so it is OK not to sync it
  empty_kernel<<<1, 1, 0, stream>>>();

#ifdef USE_CUBLAS
  cublasDestroy(handle);
#endif

  // test ptsz
  // CUstream s2;
  // cuStreamCreate(&s2, CU_STREAM_DEFAULT);
  // cuStreamSynchronize(s2);
  // cuStreamDestroy(s2);
}