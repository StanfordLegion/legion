/* Copyright 2022 Stanford University, NVIDIA Corporation
 * Copyright 2022 Los Alamos National Laboratory
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
#include "realm/cmdline.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

/**
 * This test verifies the synchronization properties that Realm::Reservation
 * provides.  It does this by creating an array of reservations that protect an
 * section of a memory resource.  Then tasks are scheduled that acquire and
 * update the associated memory resource
 */

enum TaskIds {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  MAKE_RESERVATIONS_TASK,
  CHAIN_RESERVATIONS_TASK,
  RANDOM_RESERVATIONS_TASK,
  UPDATE_VALUE_TASK,
  // Add task ids here
  NUM_TASKS
};

struct TopLevelParams {
  unsigned locks_per_processor;
  unsigned tasks_per_processor;
  TopLevelParams() : locks_per_processor(8), tasks_per_processor(4) {}
};

struct MakeReservationsParams {
  size_t idx;
  size_t stride;
  RegionInstance reservation_instance;
  RegionInstance value_instance;
};

struct ChainReservationsParams {
  size_t idx;
  size_t num_reservations;
  RegionInstance reservation_instance;
  RegionInstance value_instance;
};

struct RandomReservationsParams {
  size_t idx;
  size_t num_reservations;
  RegionInstance reservation_instance;
  RegionInstance value_instance;
};

struct UpdateValueParams {
  size_t idx;
  RegionInstance value_instance;
};

static Event get_data(RegionInstance src, Rect<1> range, size_t field_id,
                      size_t field_size, RegionInstance &dst,
                      Memory mem = Memory::NO_MEMORY) {
  Event e = Event::NO_EVENT;
  std::vector<size_t> field_sizes(1, field_size);
  std::vector<CopySrcDstField> src_desc(1), dst_desc(1);
  if (src.address_space() !=
               Processor::get_executing_processor().address_space()) {
    if (mem == Memory::NO_MEMORY) {
      mem = Machine::MemoryQuery(Machine::get_machine())
                .local_address_space()
                .has_affinity_to(Processor::get_executing_processor())
                .has_capacity(field_size * range.volume())
                .first();
    }
    src.fetch_metadata(Processor::get_executing_processor()).wait();
    e = Event::merge_events(
        e, RegionInstance::create_instance(dst, mem, range, field_sizes, 0,
                                           ProfilingRequestSet()));
    e.wait();
    src_desc[0].set_field(src, field_id, field_size);
    dst_desc[0].set_field(dst, field_id, field_size);
    e = range.copy(src_desc, dst_desc, ProfilingRequestSet());
  } else {
    dst = src;
  }
  return e;
}

static Event put_data(RegionInstance dst, Rect<1> range, size_t field_id,
                      size_t field_size, RegionInstance src,
                      Event e = Event::NO_EVENT) {
  if (dst != src) {
    std::vector<CopySrcDstField> src_desc(1), dst_desc(1);
    e = Event::merge_events(
        e, dst.fetch_metadata(Processor::get_executing_processor()));
    src_desc[0].set_field(src, field_id, field_size);
    dst_desc[0].set_field(dst, field_id, field_size);
    return range.copy(src_desc, dst_desc, ProfilingRequestSet(), e);
  } else {
    return e;
  }
}

static void top_level_task(const void *args, size_t arglen,
                           const void *userdata, size_t userlen, Processor p) {
  const struct TopLevelParams &params =
      *reinterpret_cast<const struct TopLevelParams *>(args);

  assert(arglen == sizeof(params));

  RegionInstance reservation_instance, value_instance;

  // Initialization: Have each node and processor create N reservations and
  // share them with all the other processors
  Machine::ProcessorQuery procs =
      Machine::ProcessorQuery(Machine::get_machine())
          .only_kind(Processor::LOC_PROC);

  Rect<1> domain(0, (params.locks_per_processor * procs.count()) - 1);
  IndexSpace<1> idx_space(domain);

  Memory memory = Machine::MemoryQuery(Machine::get_machine())
                      .local_address_space()
                      .has_affinity_to(p)
                      .has_capacity((sizeof(Reservation) + sizeof(size_t)) *
                                    domain.volume())
                      .first();

  std::vector<size_t> reservation_field_sizes(1, sizeof(Reservation));
  std::vector<size_t> value_field_sizes(1, sizeof(size_t));

  Event rsrv_inst_event = RegionInstance::create_instance(
      reservation_instance, memory, idx_space, reservation_field_sizes, 0,
      ProfilingRequestSet());

  Event value_inst_event = RegionInstance::create_instance(
      value_instance, memory, idx_space, reservation_field_sizes, 0,
      ProfilingRequestSet());

  {
    // Have all the processors create their reservations
    std::vector<Event> events(procs.count() + 1, Event::NO_EVENT);
    Machine::ProcessorQuery::iterator target_proc = procs.begin();
    struct MakeReservationsParams res_params;
    res_params.reservation_instance = reservation_instance;
    res_params.stride = params.locks_per_processor;
    // Have each processor make a number of reservations
    for (size_t i = 0; i < procs.count(); ++target_proc, ++i) {
      res_params.idx = i;
      events[res_params.idx] =
          target_proc->spawn(MAKE_RESERVATIONS_TASK, &res_params,
                             sizeof(res_params), rsrv_inst_event);
    }
    // Clear the value array
    size_t fill_value = 0;
    std::vector<CopySrcDstField> fill_desc(1);
    fill_desc[0].set_field(value_instance, 0, sizeof(fill_value));
    events.back() = idx_space
                        .fill(fill_desc, ProfilingRequestSet(), &fill_value,
                              sizeof(fill_value), value_inst_event);
    Event::merge_events(events).wait();
  }

  {
    log_app.info("=== Case 1: Chain ===");
    std::vector<Event> events(params.tasks_per_processor * procs.count());
    Machine::ProcessorQuery::iterator target_proc = procs.begin();
    struct ChainReservationsParams chain_params;
    chain_params.reservation_instance = reservation_instance;
    chain_params.value_instance = value_instance;
    chain_params.num_reservations = domain.volume();
    // Launch all the chaining tasks
    for (size_t i = 0; i < procs.count(); ++i, ++target_proc) {
      for (size_t j = 0; j < params.tasks_per_processor; ++j) {
        chain_params.idx = i * params.tasks_per_processor + j;
        events[chain_params.idx] = target_proc->spawn(
            CHAIN_RESERVATIONS_TASK, &chain_params, sizeof(chain_params));
      }
    }
    Event::merge_events(events).wait();
    // verify everyone incremented the array
    AffineAccessor<size_t, 1> values(value_instance, 0, 0);
    for (size_t i = 0; i < chain_params.num_reservations; i++) {
      size_t value = values[i];
      if (value != params.tasks_per_processor * procs.count()) {
        log_app.error("%s(%d): Value[%llu]=%llu", __FILE__, __LINE__, (unsigned long long)i, (unsigned long long)value);
        assert(value == params.tasks_per_processor * procs.count());
      }
    }
  }

  {
    log_app.info("=== Case 2: Random ===");
    std::vector<Event> events(params.tasks_per_processor * procs.count(), Event::NO_EVENT);
    Machine::ProcessorQuery::iterator target_proc = procs.begin();
    struct RandomReservationsParams random_params;
    random_params.reservation_instance = reservation_instance;
    random_params.value_instance = value_instance;
    random_params.num_reservations = domain.volume();
    // Launch all the chaining tasks
    for (size_t i = 0; i < procs.count(); ++i, ++target_proc) {
      for (size_t j = 0; j < params.tasks_per_processor; ++j) {
        random_params.idx = i * params.tasks_per_processor + j;
        events[random_params.idx] = target_proc->spawn(
            RANDOM_RESERVATIONS_TASK, &random_params, sizeof(random_params));
      }
    }
    Event::merge_events(events).wait();
    // verify everyone incremented the array
    AffineAccessor<size_t, 1> values(value_instance, 0, 0);
    for (size_t i = 0; i < random_params.num_reservations; i++) {
      size_t value = values[i];
      if (value != 2 * params.tasks_per_processor * procs.count()) {
        log_app.error("%s(%d): Value[%llu]=%llu", __FILE__, __LINE__, (unsigned long long)i, (unsigned long long)value);
        assert(value == 2 * params.tasks_per_processor * procs.count());
      }
    }
  }
  log_app.print("&&&& PASSED");
}

static void make_reservations_task(const void *args, size_t arglen,
                                   const void *userdata, size_t userlen,
                                   Processor p) {
  const struct MakeReservationsParams &params =
      *reinterpret_cast<const struct MakeReservationsParams *>(args);

  assert(arglen == sizeof(params));
  RegionInstance tmp_instance;
  Rect<1> update_rect = Rect<1>(params.idx * params.stride,
                                ((params.idx + 1) * params.stride) - 1);

  {
    Memory local_memory = Machine::MemoryQuery(Machine::get_machine())
                              .has_affinity_to(p)
                              .has_capacity(sizeof(Reservation) * params.stride)
                              .first();
    std::vector<size_t> tmp_inst_field_sizes(1, sizeof(Reservation));
    RegionInstance::create_instance(tmp_instance, local_memory, update_rect,
                                    tmp_inst_field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();

    AffineAccessor<Reservation, 1> reservations(tmp_instance, 0);
    for (size_t i = 0; i < params.stride; i++) {
      Reservation res = Reservation::create_reservation();
      assert(res.exists());
      reservations[i + update_rect.lo] = res;
    }
  }

  Event e = put_data(params.reservation_instance, update_rect, 0,
                     sizeof(Reservation), tmp_instance);

  e.wait();
  tmp_instance.destroy(e);
}

static void chain_reservations_task(const void *args, size_t arglen,
                                    const void *userdata, size_t userlen,
                                    Processor p) {
  const struct ChainReservationsParams &params =
      *reinterpret_cast<const struct ChainReservationsParams *>(args);

  assert(arglen == sizeof(params));

  RegionInstance local_rsrv_instance;
  Event e = get_data(params.reservation_instance,
                     Rect<1>(0, params.num_reservations - 1), 0,
                     sizeof(Reservation), local_rsrv_instance);
  e.wait();

  AffineAccessor<Reservation, 1> rsrvs =
      AffineAccessor<Reservation, 1>(local_rsrv_instance, 0, 0);

  // Pick a random place based on the parameter index
  const size_t offset = params.idx * params.num_reservations;

  for (size_t i = 0; i < params.num_reservations; i++) {
    struct UpdateValueParams update_params;
    const size_t rsrv_idx = (i + offset) % params.num_reservations;
    update_params.idx = rsrv_idx;
    update_params.value_instance = params.value_instance;
    Reservation &res = rsrvs[rsrv_idx];
    assert(res.exists());
    // Now, acquire the reservation, and while acquired, update the value
    // instance.  No other processor may update this at the same time
    e = res.acquire(0, true, e);
    e = p.spawn(UPDATE_VALUE_TASK, &update_params, sizeof(update_params), e);
    res.release(e);
  }

  if (local_rsrv_instance != params.reservation_instance) {
    local_rsrv_instance.destroy(e);
  }
  e.wait();
}

static void random_reservations_task(const void *args, size_t arglen,
                                     const void *userdata, size_t userlen,
                                     Processor p) {
  const struct RandomReservationsParams &params =
      *reinterpret_cast<const struct RandomReservationsParams *>(args);

  assert(arglen == sizeof(params));

  RegionInstance local_rsrv_instance;
  get_data(params.reservation_instance, Rect<1>(0, params.num_reservations - 1),
           0, sizeof(Reservation), local_rsrv_instance)
      .wait();

  AffineAccessor<Reservation, 1> rsrvs =
      AffineAccessor<Reservation, 1>(local_rsrv_instance, 0, 0);

  // Pick a random place to start
  const size_t offset = lrand48();
  std::vector<Event> events(params.num_reservations, Event::NO_EVENT);

  for (size_t i = 0; i < params.num_reservations; i++) {
    struct UpdateValueParams update_params;
    const size_t rsrv_idx = (i + offset) % params.num_reservations;
    update_params.idx = rsrv_idx;
    update_params.value_instance = params.value_instance;
    Reservation &res = rsrvs[rsrv_idx];
    assert(res.exists());
    // Now, acquire the reservation, and while acquired, update the value
    // instance.  No other processor may update this at the same time
    Event e = res.acquire();
    events[i] =
        p.spawn(UPDATE_VALUE_TASK, &update_params, sizeof(update_params), e);
    res.release(events[i]);
  }

  {
    Event e = Event::merge_events(events);

    if (local_rsrv_instance != params.reservation_instance) {
      local_rsrv_instance.destroy(e);
    }
    e.wait();
  }
}

static void update_value_task(const void *args, size_t arglen,
                              const void *userdata, size_t userlen,
                              Processor p) {
  const struct UpdateValueParams &params =
      *reinterpret_cast<const struct UpdateValueParams *>(args);

  assert(arglen == sizeof(struct UpdateValueParams));
  Rect<1> region_rect = Rect<1>(params.idx, params.idx);
  RegionInstance value_instance;

  get_data(params.value_instance, region_rect, 0, sizeof(size_t),
           value_instance)
      .wait();

  AffineAccessor<size_t, 1> values =
      AffineAccessor<size_t, 1>(value_instance, 0, 0);
  size_t value = values[params.idx];
  values[params.idx] = value + 1;

  Event e = put_data(params.value_instance, region_rect, 0, sizeof(size_t),
                     value_instance);

  if (value_instance != params.value_instance) {
    value_instance.destroy(e);
  }
  e.wait();
}

int main(int argc, char **argv) {
  Runtime rt;
  CommandLineParser cp;
  std::vector<Event> events;
  TopLevelParams params;

  rt.init(&argc, &argv);

  cp.add_option_int("-num_locks", params.locks_per_processor);
  cp.add_option_int("-num_tasks", params.tasks_per_processor);

  if (!cp.parse_command_line(argc, const_cast<const char **>(argv))) {
    return -1;
  }

  events.push_back(Processor::register_task_by_kind(
      Processor::LOC_PROC, false, TOP_LEVEL_TASK,
      CodeDescriptor(top_level_task), ProfilingRequestSet(), 0, 0));
  events.push_back(Processor::register_task_by_kind(
      Processor::LOC_PROC, false, MAKE_RESERVATIONS_TASK,
      CodeDescriptor(make_reservations_task), ProfilingRequestSet(), 0, 0));
  events.push_back(Processor::register_task_by_kind(
      Processor::LOC_PROC, false, CHAIN_RESERVATIONS_TASK,
      CodeDescriptor(chain_reservations_task), ProfilingRequestSet(), 0, 0));
  events.push_back(Processor::register_task_by_kind(
      Processor::LOC_PROC, false, RANDOM_RESERVATIONS_TASK,
      CodeDescriptor(random_reservations_task), ProfilingRequestSet(), 0, 0));
  events.push_back(Processor::register_task_by_kind(
      Processor::LOC_PROC, false, UPDATE_VALUE_TASK,
      CodeDescriptor(update_value_task), ProfilingRequestSet(), 0, 0));

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  Event fini = rt.collective_spawn(p, TOP_LEVEL_TASK, &params, sizeof(params),
                                   Event::merge_events(events));

  rt.shutdown(fini);

  return rt.wait_for_shutdown();
}

