/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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

// Test that GPU reductions work correctly when only the basic 4 CUDA kernels
// (apply_excl, apply_nonexcl, fold_excl, fold_nonexcl) are registered with
// register_reduction, leaving the advanced/transpose slots null.
//
// ReductionOpAdd has LHS == RHS (both uint32_t), so sizeof_lhs == sizeof_rhs
// and the fast-path condition in progress_xd is satisfied.  The
// has_fast_kernels guard (the change under test) then causes the dispatch to
// fall back to the basic stride loop rather than crashing.

#include "realm.h"
#include "realm/cmdline.h"

#include <cassert>
#include <cstdlib>

#include "cuda_reduce_basic.h"

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

enum
{
  FID_DATA = 44,
};

enum
{
  REDOP_BASIC = 99,
};

typedef ReductionOpAdd::LHS T; // uint32_t, same as RHS

// Runs apply and fold reductions with src and dst both in gpu_mem.
// Results are copied to cpu_mem for verification.
template <int N, typename PT>
bool test_reduce_dim(Rect<N, PT> domain, Memory gpu_mem, Memory cpu_mem)
{
  std::map<FieldID, size_t> fields;
  fields[FID_DATA] = sizeof(T);

  IndexSpace<N, PT> is(domain);

  RegionInstance src_inst, dst_inst, cpu_inst;
  RegionInstance::create_instance(src_inst, gpu_mem, is, fields, 0, ProfilingRequestSet())
      .wait();
  RegionInstance::create_instance(dst_inst, gpu_mem, is, fields, 0, ProfilingRequestSet())
      .wait();
  RegionInstance::create_instance(cpu_inst, cpu_mem, is, fields, 0, ProfilingRequestSet())
      .wait();

  size_t errors = 0;

  // --- apply test ---
  // src = 3, dst = 10 -> expected 10 + 3 = 13
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_fill<T>(3);
    dsts[0].set_field(src_inst, FID_DATA, sizeof(T));
    is.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_fill<T>(10);
    dsts[0].set_field(dst_inst, FID_DATA, sizeof(T));
    is.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(src_inst, FID_DATA, sizeof(T));
    dsts[0].set_field(dst_inst, FID_DATA, sizeof(T));
    dsts[0].set_redop(REDOP_BASIC, false /*!is_fold*/);
    is.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(dst_inst, FID_DATA, sizeof(T));
    dsts[0].set_field(cpu_inst, FID_DATA, sizeof(T));
    is.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
  {
    AffineAccessor<T, N, PT> acc(cpu_inst, FID_DATA);
    for(IndexSpaceIterator<N, PT> it(is); it.valid; it.step())
      for(PointInRectIterator<N, PT> it2(it.rect); it2.valid; it2.step()) {
        T actual = acc[it2.p];
        if(actual != 13) {
          if(++errors < 5)
            log_app.error() << "apply mismatch at " << it2.p << ": got " << actual
                            << " expected 13";
        }
      }
  }

  // --- fold test ---
  // src = 3, dst = 5 -> expected 5 + 3 = 8
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_fill<T>(5);
    dsts[0].set_field(dst_inst, FID_DATA, sizeof(T));
    is.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(src_inst, FID_DATA, sizeof(T));
    dsts[0].set_field(dst_inst, FID_DATA, sizeof(T));
    dsts[0].set_redop(REDOP_BASIC, true /*is_fold*/);
    is.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(dst_inst, FID_DATA, sizeof(T));
    dsts[0].set_field(cpu_inst, FID_DATA, sizeof(T));
    is.copy(srcs, dsts, ProfilingRequestSet()).wait();
  }
  {
    AffineAccessor<T, N, PT> acc(cpu_inst, FID_DATA);
    for(IndexSpaceIterator<N, PT> it(is); it.valid; it.step())
      for(PointInRectIterator<N, PT> it2(it.rect); it2.valid; it2.step()) {
        T actual = acc[it2.p];
        if(actual != 8) {
          if(++errors < 5)
            log_app.error() << "fold mismatch at " << it2.p << ": got " << actual
                            << " expected 8";
        }
      }
  }

  src_inst.destroy();
  dst_inst.destroy();
  cpu_inst.destroy();

  return (errors == 0);
}

void top_level_task(const void *data, size_t datalen, const void *userdata,
                    size_t userlen, Processor p)
{
  Memory gpu_mem = Memory::NO_MEMORY;
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::GPU_FB_MEM).has_capacity(1);
  for(Memory m : mq) {
    gpu_mem = m;
    break;
  }

  if(!gpu_mem.exists()) {
    log_app.warning() << "no GPU FB memory found, skipping test";
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 0);
    return;
  }

  Memory cpu_mem =
      Machine::MemoryQuery(Machine::get_machine()).has_affinity_to(p).first();
  assert(cpu_mem.exists());

  bool ok = true;

  if(ok) {
    Rect<1> r(0, 15);
    ok = test_reduce_dim(r, gpu_mem, cpu_mem);
    if(!ok)
      log_app.error() << "1-D basic reduce test FAILED";
  }

  if(ok) {
    Rect<2> r(Point<2>(0, 0), Point<2>(7, 7));
    ok = test_reduce_dim(r, gpu_mem, cpu_mem);
    if(!ok)
      log_app.error() << "2-D basic reduce test FAILED";
  }

  if(ok) {
    Rect<3> r(Point<3>(0, 0, 0), Point<3>(3, 3, 3));
    ok = test_reduce_dim(r, gpu_mem, cpu_mem);
    if(!ok)
      log_app.error() << "3-D basic reduce test FAILED";
  }

  if(ok)
    log_app.info() << "all basic reduce tests PASSED";

  Runtime::get_runtime().shutdown(Event::NO_EVENT, ok ? 0 : 1);
}

extern void register_basic_gpu_reduction(Realm::Runtime &realm,
                                         Realm::ReductionOpID redop_id);

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  register_basic_gpu_reduction(rt, REDOP_BASIC);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  return rt.wait_for_shutdown();
}
