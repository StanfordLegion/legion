/* Copyright 2017 Stanford University
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

#include "simple_blas.h"

using namespace LegionRuntime::Arrays;
using namespace LegionRuntime::Accessor;

// global configuration variables
int blas_thread_count = 0;
bool blas_do_parallel = true;

// single-precision float

/*extern*/ BlasTaskImplementations<float> blas_impl_s;

template <>
void BlasTaskImplementations<float>::axpy_task_cpu(const Task *task,
						   const std::vector<PhysicalRegion> &regions,
						   Context ctx, Runtime *runtime)
{
  IndexSpace is = regions[1].get_logical_region().get_index_space();
  Rect<1> bounds = runtime->get_index_space_domain(ctx, is).get_rect<1>();
  //printf("hi [%d, %d]\n", bounds.lo[0], bounds.hi[0]);

  float alpha = *(const float *)(task->args);

  RegionAccessor<AccessorType::Affine<1>, float> fa_x = regions[0].get_field_accessor(task->regions[0].instance_fields[0]).typeify<float>().convert<AccessorType::Affine<1> >();
  RegionAccessor<AccessorType::Affine<1>, float> fa_y = regions[1].get_field_accessor(task->regions[1].instance_fields[0]).typeify<float>().convert<AccessorType::Affine<1> >();

#pragma omp parallel for if(blas_do_parallel)
  for(int i = bounds.lo[0]; i <= bounds.hi[0]; i++)
    fa_y[i] += alpha * fa_x[i];
}

template <>
float BlasTaskImplementations<float>::dot_task_cpu(const Task *task,
						   const std::vector<PhysicalRegion> &regions,
						   Context ctx, Runtime *runtime)
{
  IndexSpace is = regions[1].get_logical_region().get_index_space();
  Rect<1> bounds = runtime->get_index_space_domain(ctx, is).get_rect<1>();
  //printf("hi [%d, %d]\n", bounds.lo[0], bounds.hi[0]);

  RegionAccessor<AccessorType::Affine<1>, float> fa_x = regions[0].get_field_accessor(task->regions[0].instance_fields[0]).typeify<float>().convert<AccessorType::Affine<1> >();
  RegionAccessor<AccessorType::Affine<1>, float> fa_y = regions[1].get_field_accessor(task->regions[1].instance_fields[0]).typeify<float>().convert<AccessorType::Affine<1> >();

  float acc = 0;
#pragma omp parallel for reduction(+:acc) if(blas_do_parallel)
  for(int i = bounds.lo[0]; i <= bounds.hi[0]; i++)
    acc += fa_x[i] * fa_y[i];
  return acc;
}
