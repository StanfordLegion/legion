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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <legion.h>
#include <realm/cmdline.h>

#include "simple_blas.h"

using namespace Legion;

Logger log_app("app");

enum {
  TOP_LEVEL_TASK_ID,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  size_t num_elements = 32768;
  size_t num_pieces = 1;

  const InputArgs &command_args = Runtime::get_input_args();

  bool ok = Realm::CommandLineParser()
    .add_option_int("-n", num_elements)
    .add_option_int("-p", num_pieces)
    .parse_command_line(command_args.argc, (const char **)command_args.argv);

  if(!ok) {
    log_app.fatal() << "error parsing command line arguments";
    exit(1);
  }

  Rect<1> r_top = Rect<1>(0, num_elements - 1);
  IndexSpace is_top = runtime->create_index_space(ctx, r_top);

  IndexPartition ip_dist = IndexPartition::NO_PART;
  if(num_pieces > 1) {
    Rect<1> cspace = Rect<1>(1, num_pieces);
    IndexSpace cs = runtime->create_index_space(ctx, cspace);
    ip_dist = runtime->create_equal_partition(ctx, is_top, cs);
  }

  BlasArrayRef<float> x = BlasArrayRef<float>::create(runtime, ctx, is_top);
  BlasArrayRef<float> y = BlasArrayRef<float>::create(runtime, ctx, is_top);

  x.fill(runtime, ctx, 2.0);
  y.fill(runtime, ctx, 3.0);

  axpy(runtime, ctx, 0.5f, x, y, ip_dist);

  float result = dot(runtime, ctx, x, y, ip_dist);
  float exp_result = num_elements * 2.0 * (3.0 + 0.5 * 2.0);

  printf("got %g, exp %g\n", result, exp_result);
}

int main(int argc, char **argv)
{
  if(getenv("OMP_SAXPY_THREADS"))
    blas_thread_count = atoi(getenv("OMP_SAXPY_THREADS"));
  if(getenv("OMP_SAXPY_SERIAL"))
    blas_do_parallel = false;

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  blas_impl_s.preregister_tasks();

  return Runtime::start(argc, argv);
}

