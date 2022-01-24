/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

// Create a random number generator functor
struct RandomGenerator {
  double start, stop;

  __host__ __device__
  RandomGenerator(double a, double b) : start(a), stop(b) { };

  __host__ __device__
  double operator()(const unsigned int n) const
  {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<double> dist(start, stop);
    rng.discard(n);
    return dist(rng);
  }
};

__host__
void initialize_gpu_array(double *ptr, size_t size)
{
  // Make our device pointers 
  thrust::device_ptr<double> device_ptr(ptr);
  // Fill the vector with random numbers between 0.0 and 1.0
  thrust::counting_iterator<unsigned int> index_sequence(0);
  thrust::transform(index_sequence, index_sequence + size,
                    device_ptr, RandomGenerator(0.0, 1.0));
}

// Make the saxpy functor
struct SaxpyFunctor : public thrust::binary_function<double,double,double> {
  const double alpha;

  __host__ __device__
  SaxpyFunctor(double a) : alpha(a) { }

  __host__ __device__
  double operator()(const double &x, const double &y) const
  {
    return alpha * x + y;
  }
};

__host__
void gpu_saxpy(double alpha, const double *x_ptr, 
               const double *y_ptr, double *z_ptr, size_t size)
{
  // Make our device pointers 
  thrust::device_ptr<double> x_vec(const_cast<double*>(x_ptr));
  thrust::device_ptr<double> y_vec(const_cast<double*>(y_ptr));
  thrust::device_ptr<double> z_vec(z_ptr);

  thrust::transform(x_vec, x_vec + size, y_vec, z_vec, SaxpyFunctor(alpha));
}

