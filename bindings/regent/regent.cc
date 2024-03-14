/* Copyright 2024 Stanford University
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

#include <cfloat>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <limits>
#include <vector>
#include <cinttypes>

#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"
#include "legion/legion_c_util.h"
#include "legion/legion_redop.h"
#include "realm/redop.h"

#include "regent.h"
#include "regent_redop.h"

using namespace Legion;

typedef Realm::Point<1,coord_t> Point1D;
typedef Realm::Point<2,coord_t> Point2D;
typedef Realm::Point<3,coord_t> Point3D;
typedef CObjectWrapper::ArrayAccessor1D ArrayAccessor1D;
typedef CObjectWrapper::ArrayAccessor2D ArrayAccessor2D;
typedef CObjectWrapper::ArrayAccessor3D ArrayAccessor3D;

#if !defined(LEGION_USE_CUDA) && !defined(LEGION_USE_HIP)
// declare (CPU-only) reductions here

#define DECLARE_ARRAY_REDUCTION(REG, CLASS)                             \
  extern "C"                                                                 \
  {                                                                          \
    void REG(legion_reduction_op_id_t redop_id, unsigned array_size,         \
             bool permit_duplicates)                                         \
    {                                                                        \
      ArrayReductionOp<CLASS> *op = ArrayReductionOp<CLASS>::create_reduction_op(array_size); \
      Runtime::register_reduction_op(redop_id, op, NULL, NULL,               \
                                     permit_duplicates);                     \
    }                                                                        \
  }

REGENT_ARRAY_REDUCE_LIST(DECLARE_ARRAY_REDUCTION)

#undef DECLARE_ARRAY_REDUCTION

#endif

static std::set<int64_t> registered_kernel_ids;

void regent_register_kernel_id(int64_t kernel_id)
{
  if (registered_kernel_ids.find(kernel_id) != registered_kernel_ids.end())
  {
    fprintf(stderr,
      "Some other CUDA kernel has already been registered with ID %" PRId64 ". "
      "This is a Regent compiler bug. Please report this on GibHub.",
      kernel_id);
    exit(-1);
  }
  registered_kernel_ids.insert(kernel_id);
}
