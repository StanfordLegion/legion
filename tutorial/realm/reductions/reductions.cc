/* Copyright 2023 Stanford University, NVIDIA Corporation
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
#include <time.h>

#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "realm.h"
#include "realm/cmdline.h"
#include "realm/id.h"

using namespace Realm;

Logger log_app("app");

enum {
  FID_INT = 44,
  FID_COMPLEX_TYPE = 45,
};

enum {
  REDOP_SUM = 99,
};

namespace TestConfig {
bool all_memories = false;
};

struct StructType {
  int value;
  bool operator!=(const StructType &other) const {
    return value != other.value;
  }
};

class SumReduction {
 public:
  using LHS = StructType;
  using RHS = int;

  template <bool EXCLUSIVE>
  static void apply(LHS &lhs, RHS rhs) {
    assert(EXCLUSIVE);
    lhs.value += rhs;
  }

  static const RHS identity;
  template <bool EXCLUSIVE>
  static void fold(RHS &rhs1, RHS rhs2) {
    if (EXCLUSIVE)
      rhs1 += rhs2;
    else {
    __sync_fetch_and_add(&rhs1, rhs2);
    }
  }
};

template <int N, typename T, typename FID_T>
bool verify(RegionInstance dst_inst, IndexSpace<N, T> domain, FieldID dst_fid,
            FID_T exp) {
  size_t errors = 0;
  AffineAccessor<FID_T, N, T> acc(dst_inst, dst_fid);
  for (IndexSpaceIterator<N, T> it(domain); it.valid; it.step()) {
    for (PointInRectIterator<N, T> it2(it.rect); it2.valid; it2.step()) {
      FID_T act = acc[it2.p];
      if (act != exp) {
        if (++errors < 10) log_app.error() << "mismatch: [" << it2.p << "]";
      }
    }
  }
  return errors == 0;
}

template <int N, typename T, typename FID_T>
Event reduce(RegionInstance src_inst, RegionInstance dst_inst,
             IndexSpace<N, T> domain, FieldID src_fid, FieldID dst_fid,
             size_t src_fsize, size_t dst_fsize, bool fold,
             bool exclusive, Event wait_on = Event::NO_EVENT) {
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_field(src_inst, src_fid, src_fsize);
  dsts[0].set_field(dst_inst, dst_fid, dst_fsize);
  dsts[0].set_redop(REDOP_SUM, fold, exclusive);
  return domain.copy(srcs, dsts, ProfilingRequestSet(), wait_on);
}

template <int N, typename T, typename FID_T>
Event fill(RegionInstance inst, IndexSpace<N, T> bloat, FieldID fid, FID_T val,
           Event wait_on = Event::NO_EVENT) {
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_fill<FID_T>(val);
  dsts[0].set_field(inst, fid, sizeof(T));
  return bloat.copy(srcs, dsts, ProfilingRequestSet(), wait_on);
}

const SumReduction::RHS SumReduction::identity = 0;

template <int N, typename T>
bool do_reduction(IndexSpace<N, T> domain, IndexSpace<N, T> bloat,
                    Memory dst_mem, Memory src_mem) {
  std::map<FieldID, size_t> fields;
  fields[FID_INT] = sizeof(int);
  fields[FID_COMPLEX_TYPE] = sizeof(StructType);

  UserEvent start_event = UserEvent::create_user_event();

  RegionInstance src_inst, dst_inst0, dst_inst1;
  Event inst_event1 = RegionInstance::create_instance(
      dst_inst0, dst_mem, bloat, fields,
      /*block_size=*/0, ProfilingRequestSet(), start_event);

  Event inst_event2 = RegionInstance::create_instance(
      dst_inst1, dst_mem, bloat, fields,
      /*block_size=*/0, ProfilingRequestSet(), start_event);

  Event inst_event3 = RegionInstance::create_instance(
      src_inst, src_mem, bloat, fields,
      /*block_size=*/0, ProfilingRequestSet(), start_event);

  std::vector<Event> fill_events;
  fill_events.push_back(
      fill(src_inst, bloat, /*fid=*/FID_INT, /*val=*/1, inst_event1));
  fill_events.push_back(
      fill(dst_inst0, bloat, /*fid=*/FID_INT, /*val=*/3, inst_event2));
  fill_events.push_back(fill(dst_inst1, bloat, /*fid=*/FID_COMPLEX_TYPE,
                             /*val=*/StructType{1}, inst_event3));

  Event fill_event = Event::merge_events(fill_events);

  // Run non-exclusive folds with atomics.
  Event fold_event1 = reduce<2, T, int>(src_inst, dst_inst0, domain,
                                        /*src_fid=*/FID_INT,
                                        /*dst_fid=*/FID_INT,
                                        /*src_fsize=*/sizeof(int),
                                        /*dst_fsize*/ sizeof(int),
                                        /*exclusive=*/false,
                                        /*fold=*/true, fill_event);

  Event fold_event2 = reduce<2, T, int>(src_inst, dst_inst0, domain,
                                        /*src_fid=*/FID_INT,
                                        /*dst_fid=*/FID_INT,
                                        /*src_fsize=*/sizeof(int),
                                        /*dst_fsize*/ sizeof(int),
                                        /*exclusive=*/false,
                                        /*fold=*/true, fill_event);

  Event fold_event = Event::merge_events(fold_event1, fold_event2);

  // Run exclusive applies with reservations.
  Reservation resrv = Reservation::create_reservation();

  Event resrv_event1 = resrv.acquire(0, true, fold_event);
  Event apply_event1 =
      reduce<2, T, StructType>(dst_inst0, dst_inst1, domain,
                               /*src_fid=*/FID_INT,
                               /*dst_fid=*/FID_COMPLEX_TYPE,
                               /*src_fsize=*/sizeof(int),
                               /*dst_fsize=*/sizeof(StructType),
                               /*exclusive=*/true,
                               /*fold=*/false, resrv_event1);
  resrv.release(apply_event1);

  Event resrv_event2 = resrv.acquire(0, true, fold_event);
  Event apply_event2 =
      reduce<2, T, StructType>(dst_inst0, dst_inst1, domain,
                               /*src_fid=*/FID_INT,
                               /*dst_fid=*/FID_COMPLEX_TYPE,
                               /*src_fsize=*/sizeof(int),
                               /*dst_fsize=*/sizeof(StructType),
                               /*exclusive=*/true,
                               /*fold=*/false, resrv_event2);
  resrv.release(apply_event2);

  start_event.trigger();
  Event::merge_events(apply_event1, apply_event2).wait();

  // Verify the results.
  bool success = verify<2, T, int>(dst_inst0, domain,
                                   /*dst_fid=*/FID_INT,
                                   /*exp=*/5) &&
                 verify<2, T, StructType>(dst_inst1, domain,
                                          /*dst_fid=*/FID_COMPLEX_TYPE,
                                          /*exp=*/StructType{11});

  dst_inst1.destroy();
  dst_inst0.destroy();
  src_inst.destroy();
  return success;
}

int main(int argc, char **argv) {
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_bool("-all", TestConfig::all_memories);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_reduction<SumReduction>(REDOP_SUM);

  std::vector<Memory> mems;
  Machine::MemoryQuery mq(Machine::get_machine());
  for (Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it)
    if ((*it).capacity() > 0) mems.push_back(*it);

  bool success = true;
  for (size_t i = 0; i < mems.size(); i++) {
    if (success) {
      success =
          do_reduction(IndexSpace<2>(Rect<2>(Point<2>(0, 0), Point<2>(3, 0))),
                       IndexSpace<2>(Rect<2>(Point<2>(0, 0), Point<2>(7, 0))),
                       mems[i], mems[i]);
    }
  }

  Runtime::get_runtime().shutdown(Event::NO_EVENT, success ? 0 : 1);
  return rt.wait_for_shutdown();
}


