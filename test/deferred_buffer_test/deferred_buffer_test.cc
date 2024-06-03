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

// Unit test exercising various constructors of DeferredBuffer

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include "legion.h"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

#define DIM LEGION_MAX_DIM

using VAL = int64_t;
using PT = int32_t;

using Accessor =
  FieldAccessor<READ_ONLY, VAL, DIM, PT, Realm::AffineAccessor<VAL, DIM, PT>>;

void validate(const Rect<DIM, PT> &rect,
              DeferredBuffer<VAL, DIM, PT> &buf,
              const VAL& init_value,
              std::array<DimensionKind, DIM> ordering)
{
  Accessor accessor(buf);

  for (PointInRectIterator<DIM, PT> itr(rect); itr(); ++itr)
    assert(accessor[*itr] == init_value);

  size_t strides[DIM];
  accessor.ptr(rect, strides);

  std::array<int32_t, DIM> dim_order;
  for (int32_t i = 0; i < DIM; ++i)
    dim_order[i] =
      static_cast<int32_t>(ordering[i]) - static_cast<int32_t>(LEGION_DIM_X);

  for (int32_t i = 0; i < DIM - 1; ++i)
    assert(strides[dim_order[i]] <= strides[dim_order[i + 1]]);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  Machine machine = Machine::get_machine();
  Machine::MemoryQuery query(machine);
  Memory memory =
    query.only_kind(Memory::Kind::SYSTEM_MEM)
         .has_affinity_to(runtime->get_executing_processor(ctx))
         .first();
  Memory::Kind kind = Memory::Kind::SYSTEM_MEM;


  Rect<DIM, PT> rect(Point<DIM, PT>::ZEROES(), Point<DIM, PT>::ONES());
  Domain bounds = rect;

  VAL init_value = 12345;
  std::array<DimensionKind, DIM> ordering;
  for (int i = 0; i < DIM; ++i)
    ordering[i] =
      static_cast<DimensionKind>(
        static_cast<int>(LEGION_DIM_X) + DIM - (i + 1));
  {
    DeferredBuffer<VAL, DIM, PT> buf(kind, bounds, &init_value, 16, false);
    validate(rect, buf, init_value, ordering);
  }
  {
    DeferredBuffer<VAL, DIM, PT> buf(rect, kind, &init_value, 16, false);
    validate(rect, buf, init_value, ordering);
  }
  for (int i = 0; i < DIM; ++i)
    ordering[i] =
      static_cast<DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
  {
    DeferredBuffer<VAL, DIM, PT> buf(memory, bounds, &init_value, 16, true);
    validate(rect, buf, init_value, ordering);
  }
  {
    DeferredBuffer<VAL, DIM, PT> buf(rect, memory, &init_value, 16, true);
    validate(rect, buf, init_value, ordering);
  }

  for (int k = 1; k < DIM; ++k)
  {
    for (int i = 0; i < DIM; ++i)
      ordering[i] =
        static_cast<DimensionKind>(
          static_cast<int>(LEGION_DIM_X) + (i + k) % DIM);
    {
      DeferredBuffer<VAL, DIM, PT> buf(memory, bounds, ordering, &init_value, 16);
      validate(rect, buf, init_value, ordering);
    }
    {
      DeferredBuffer<VAL, DIM, PT> buf(rect, memory, ordering, &init_value, 16);
      validate(rect, buf, init_value, ordering);
    }
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

  return Runtime::start(argc, argv);
}
