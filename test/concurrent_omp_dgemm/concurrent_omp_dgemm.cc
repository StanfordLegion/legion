/* Copyright 2023 Stanford University
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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#ifndef BLAS_USE_MKL
#include <cblas.h>
#else
#include <mkl.h>
#endif
#include <sys/time.h>
#include "legion.h"
using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_MATRIX_TASK_ID,
  DGEMM_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_A,
  FID_B,
  FID_C,
};

struct dgemm_task_args {
  double alpha = 1.0;
  double beta = 1.0;
  int iterations = 100;
  int num_omp_threads;
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int N = 4; 
  int num_dgemms = 4;
  dgemm_task_args dgemm_args;

  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        N = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_dgemms = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-i"))
        dgemm_args.iterations = atoi(command_args.argv[++i]);
    }
  }
  printf("Running %d DGEMMs of size N = %d...\n", num_dgemms, N);

  Realm::Runtime rt = Realm::Runtime::get_runtime();
  Realm::ModuleConfig* openmp_config = rt.get_module_config("openmp");
  assert (openmp_config != nullptr);
  assert(openmp_config->get_property("othr", dgemm_args.num_omp_threads) == true);

  // Create our logical regions using the same schemas as earlier examples
  int num_elements = N * N * num_dgemms;
  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect); 
  runtime->attach_name(is, "is");
  FieldSpace input_fs = runtime->create_field_space(ctx);
  runtime->attach_name(input_fs, "input_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_A);
    runtime->attach_name(input_fs, FID_A, "A");
    allocator.allocate_field(sizeof(double),FID_B);
    runtime->attach_name(input_fs, FID_B, "B");
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  runtime->attach_name(output_fs, "output_fs");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(double),FID_C);
    runtime->attach_name(output_fs, FID_C, "C");
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);
  runtime->attach_name(output_lr, "output_lr");

  Rect<1> color_bounds(0, num_dgemms-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");
  
  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
  runtime->attach_name(output_lp, "output_lp");

  ArgumentMap arg_map;

  IndexLauncher init_launcher(INIT_MATRIX_TASK_ID, color_is,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_A);
  init_launcher.region_requirements[0].add_field(FID_B);
  init_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  init_launcher.region_requirements[1].add_field(FID_C);
  runtime->execute_index_space(ctx, init_launcher);

  {
    IndexLauncher daxpy_launcher(DGEMM_TASK_ID, color_is,
                  TaskArgument(&dgemm_args, sizeof(dgemm_task_args)), arg_map);
    daxpy_launcher.add_region_requirement(
        RegionRequirement(input_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, input_lr));
    daxpy_launcher.region_requirements[0].add_field(FID_A);
    daxpy_launcher.region_requirements[0].add_field(FID_B);
    daxpy_launcher.add_region_requirement(
        RegionRequirement(output_lp, 0/*projection ID*/,
                          READ_WRITE, EXCLUSIVE, output_lr));
    daxpy_launcher.region_requirements[1].add_field(FID_C);
    daxpy_launcher.must_parallelism = true;
    runtime->execute_index_space(ctx, daxpy_launcher);
  }
                      
  IndexLauncher check_launcher(CHECK_TASK_ID, color_is,
                TaskArgument(&dgemm_args, sizeof(dgemm_task_args)), arg_map);
  check_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_A);
  check_launcher.region_requirements[0].add_field(FID_B);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_C);
  runtime->execute_index_space(ctx, check_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
  runtime->destroy_index_space(ctx, color_is);
}

void init_matrix_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2); 
  assert(task->regions.size() == 2);

  const int point = task->index_point.point_data[0];
  printf("Initializing matrix for block %d...\n", point);

  const FieldAccessor<WRITE_DISCARD,double,1> acc_a(regions[0], FID_A);
  const FieldAccessor<WRITE_DISCARD,double,1> acc_b(regions[0], FID_B);
  const FieldAccessor<WRITE_DISCARD,double,1> acc_c(regions[1], FID_C);
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    acc_a[*pir] = 1.0;
    acc_b[*pir] = 1.0;
    acc_c[*pir] = 1.0;
  }
}

void dgemm_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(dgemm_task_args));
  const dgemm_task_args dgemm_args = *((const dgemm_task_args*)task->args);
  const int point = task->index_point.point_data[0];

  const FieldAccessor<READ_ONLY,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc_a(regions[0], FID_A);
  const FieldAccessor<READ_ONLY,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc_b(regions[0], FID_B);
  const FieldAccessor<READ_WRITE,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc_c(regions[1], FID_C);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  assert(acc_a.accessor.is_dense_arbitrary(rect));
  assert(acc_b.accessor.is_dense_arbitrary(rect));
  assert(acc_c.accessor.is_dense_arbitrary(rect));

  const double *A = acc_a.ptr(rect.lo);
  const double *B = acc_b.ptr(rect.lo);
  double *C = acc_c.ptr(rect.lo);

  int N = (int)sqrt(rect.volume());

  printf("Running dgemm computation with alpha %.8g for point %d, N %d, on proc %llx, iterations %d\n", 
          dgemm_args.alpha, point, N, task->current_proc.id, dgemm_args.iterations);

  omp_set_num_threads(dgemm_args.num_omp_threads);
#ifdef BLAS_USE_OPENBLAS
  openblas_set_num_threads(dgemm_args.num_omp_threads);
#endif
#ifdef BLAS_USE_BLIS
  bli_thread_set_num_threads(dgemm_args.num_omp_threads);
#endif

  // warm up
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
            N, N, N, dgemm_args.alpha, 
            A, N, 
            B, N, 
            dgemm_args.beta, C, N);

  struct timeval tv;
  gettimeofday(&tv,NULL);
  double start_time_in_micros = 1000000 * tv.tv_sec + tv.tv_usec;
  for (int i = 0; i < dgemm_args.iterations; i++) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                N, N, N, dgemm_args.alpha, 
                A, N, 
                B, N, 
                dgemm_args.beta, C, N);
  }
  gettimeofday(&tv,NULL);
  double end_time_in_micros = 1000000 * tv.tv_sec + tv.tv_usec;

  double duration = (end_time_in_micros - start_time_in_micros) / (double)(dgemm_args.iterations);

  double flops = 2 * (double)N * (double)N * (double)N / duration / 1e3;
  printf("point %d, time %f us, gflops %f\n", point, duration, flops);
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(dgemm_task_args));
  const dgemm_task_args dgemm_args = *((const dgemm_task_args*)task->args);

  const FieldAccessor<READ_ONLY,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc_a(regions[0], FID_A);
  const FieldAccessor<READ_ONLY,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc_b(regions[0], FID_B);
  const FieldAccessor<READ_ONLY,double,1,coord_t,
          Realm::AffineAccessor<double,1,coord_t> > acc_c(regions[1], FID_C);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  assert(acc_a.accessor.is_dense_arbitrary(rect));
  assert(acc_b.accessor.is_dense_arbitrary(rect));
  assert(acc_c.accessor.is_dense_arbitrary(rect));

  const double *A = acc_a.ptr(rect.lo);
  const double *B = acc_b.ptr(rect.lo);
  const double *C = acc_c.ptr(rect.lo);

  int N = (int)sqrt(rect.volume());

  printf("Checking results...\n");

  bool all_passed = true;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double expected = 1.0;
      double received = C[i*N+j];
      for (int k = 0; k < N; k++) {
        // there is a warmup, so total number of dgemms is iterations+1
        expected += dgemm_args.alpha * A[i*N+k] * B[k*N+j] * (dgemm_args.iterations+1);
      }
      if (expected != received) {
        all_passed = false;
        printf("expected %f, received %f\n", expected, received);
      }
    }
  }

  if (all_passed)
    printf("SUCCESS!\n");
  else {
    printf("FAILURE!\n");
    abort();
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_MATRIX_TASK_ID, "init_matrix");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_matrix_task>(registrar, "init_matrix");
  }

  {
    TaskVariantRegistrar registrar(DGEMM_TASK_ID, "dgemm");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dgemm_task>(registrar, "dgemm");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
