/* Copyright 2022 Stanford University
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

// global configuration variables
int blas_thread_count = 0;
bool blas_do_parallel = true;

extern "C" {
  int omp_get_thread_num(void);
};

extern Realm::Logger log_app;

// single-precision float

/*extern*/ BlasTaskImplementations<float> blas_impl_s;

template <>
void BlasTaskImplementations<float>::axpy_task_cpu(const Task *task,
						   const std::vector<PhysicalRegion> &regions,
						   Context ctx, Runtime *runtime)
{
  IndexSpace is = regions[1].get_logical_region().get_index_space();
  Rect<1> bounds = runtime->get_index_space_domain(ctx, is);
  //printf("hi [%d, %d]\n", bounds.lo[0], bounds.hi[0]);

  float alpha = *(const float *)(task->args);

  const FieldAccessor<READ_ONLY,float,1,coord_t,
          Realm::AffineAccessor<float,1,coord_t> > fa_x(regions[0], task->regions[0].instance_fields[0]);
  const FieldAccessor<READ_WRITE,float,1,coord_t,
          Realm::AffineAccessor<float,1,coord_t> > fa_y(regions[1], task->regions[1].instance_fields[0]);

#pragma omp parallel
  {
    log_app.print() << "saxpy on proc=" << Realm::Processor::get_executing_processor() << " thread=" << omp_get_thread_num();
  }

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
  Rect<1> bounds = runtime->get_index_space_domain(ctx, is);
  //printf("hi [%d, %d]\n", bounds.lo[0], bounds.hi[0]);

  const FieldAccessor<READ_ONLY,float,1,coord_t,
          Realm::AffineAccessor<float,1,coord_t> > fa_x(regions[0], task->regions[0].instance_fields[0]);
  const FieldAccessor<READ_ONLY,float,1,coord_t,
          Realm::AffineAccessor<float,1,coord_t> > fa_y(regions[1], task->regions[0].instance_fields[0]);

  float acc = 0;
#pragma omp parallel for reduction(+:acc) if(blas_do_parallel)
  for(int i = bounds.lo[0]; i <= bounds.hi[0]; i++)
    acc += fa_x[i] * fa_y[i];
  return acc;
}
