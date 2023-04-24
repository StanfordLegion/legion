/* Copyright 2023 NVIDIA Corporation
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

#include <cmath>

using namespace Realm;

Logger log_app("app");

enum {
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  ADD_TASK,
  INIT_TASK,
  VERIFY_TASK,
};

enum {
  FID_DATA = 100,
};

namespace TestConfig {
  int num_iters = 10;
  int num_elements = 10;
};

struct VerifyTaskArgs {
  IndexSpace<1> is;
  RegionInstance inst;
  double val;
};

struct InitTaskArgs {
  IndexSpace<1> is;
  RegionInstance inst_in;
  RegionInstance inst_out; // inst_out = inst_in
};

struct AddTaskArgs {
  IndexSpace<1> is;
  RegionInstance inst_in1;
  RegionInstance inst_in2;
  RegionInstance inst_out; // inst_out = inst_in1 + inst_in2
};

void add_task(const void *args, size_t arglen, 
              const void *userdata, size_t userlen, Processor p)
{
  const AddTaskArgs& task_args = *reinterpret_cast<const AddTaskArgs *>(args);
  AffineAccessor<double, 1> acc_in1(task_args.inst_in1, FID_DATA);
  AffineAccessor<double, 1> acc_in2(task_args.inst_in2, FID_DATA);
  AffineAccessor<double, 1> acc_out(task_args.inst_out, FID_DATA);
  for(IndexSpaceIterator<1> it(task_args.is); it.valid; it.step()) {
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      acc_out[it2.p] = (acc_in1[it2.p] + acc_in2[it2.p]) /10.0 + 1.0;
    }
  }
  log_app.info() << "add: " << acc_out[0];
}

void init_task(const void *args, size_t arglen, 
               const void *userdata, size_t userlen, Processor p)
{
  const InitTaskArgs& task_args = *reinterpret_cast<const InitTaskArgs *>(args);
  AffineAccessor<double, 1> acc_in(task_args.inst_in, FID_DATA);
  AffineAccessor<double, 1> acc_out(task_args.inst_out, FID_DATA);
  for(IndexSpaceIterator<1> it(task_args.is); it.valid; it.step()) {
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      acc_out[it2.p] = acc_in[it2.p];
    }
  }
  log_app.info() << "init: " << acc_out[0];
}

void verify_task(const void *args, size_t arglen, 
                 const void *userdata, size_t userlen, Processor p)
{
  const VerifyTaskArgs& task_args = *reinterpret_cast<const VerifyTaskArgs *>(args);
  AffineAccessor<double, 1> acc(task_args.inst, FID_DATA);
  for(IndexSpaceIterator<1> it(task_args.is); it.valid; it.step()) {
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      if (acc[it2.p] != task_args.val) {
        log_app.error() << "MISMATCH: " << it2.p << ": " << acc[it2.p] << " != " << task_args.val;
      }
    }
  }
}

void main_task(const void *args, size_t arglen, const void *userdata,
               size_t userlen, Processor p) 
{
  IndexSpace<1> is = Rect<1>(0, TestConfig::num_elements-1);
  RegionInstance insts[5];
  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_DATA] = sizeof(double);

  Memory m = Machine::MemoryQuery(Machine::get_machine()).only_kind(Memory::Kind::SYSTEM_MEM).first();
  assert(m.exists());

  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::Kind::LOC_PROC);
  if (pq.count() < 3) {
    log_app.warning("The optimal number of CPU processor for this program is at least 3, please specify it through -ll:cpu");
  }
  std::vector<Processor> cpus(pq.begin(), pq.end());
  // p1 is used to run tasks on the left side of the DAG
  // p2 is used to run tasks on the right side of the DAG
  Processor p1 = Processor::NO_PROC;
  Processor p2 = Processor::NO_PROC;
  if (pq.count() == 1) {
    p1 = cpus[0];
    p2 = cpus[0];
  } else if (pq.count() == 2) {
    p1 = cpus[0];
    p2 = cpus[1];
  } else {
    p1 = cpus[1];
    p2 = cpus[2];
  }

  // create instances
  for (int i = 0; i < 5; i++) {
    RegionInstance::create_instance(insts[i], m, is, field_sizes,
                0 /*block_size=SOA*/,
                ProfilingRequestSet()).wait();
    // we set it to 1.25 because the result of add_task is (1.25+1.25) / 10 +1 = 1.25, 
    // so we can run as many tasks as we want without overflow the double
    double fill_value = 1.25;
    std::vector<CopySrcDstField> sdf(1);
    sdf[0].inst = insts[i];
    sdf[0].field_id = FID_DATA;
    sdf[0].size = sizeof(double);
    is.fill(sdf, ProfilingRequestSet(), &fill_value, sizeof(fill_value)).wait();
    VerifyTaskArgs args;
    args.is = is;
    args.val = 1.25;
    args.inst = insts[i];
    // warm up both p1 and p2
    if (i % 2 == 0) {
      p1.spawn(VERIFY_TASK, &args, sizeof(args)).wait();
    } else {
      p2.spawn(VERIFY_TASK, &args, sizeof(args)).wait();
    }
  }

  InitTaskArgs init_args1, init_args2;
  init_args1.is = is;
  init_args1.inst_in = insts[4];
  init_args1.inst_out = insts[0];
  init_args2.is = is;
  init_args2.inst_in = insts[4];
  init_args2.inst_out = insts[1];
  
  AddTaskArgs add_args1;
  add_args1.is = is;
  add_args1.inst_in1 = insts[0];
  add_args1.inst_in2 = insts[1];
  add_args1.inst_out = insts[2];

  AddTaskArgs add_args2;
  add_args2.is = is;
  add_args2.inst_in1 = insts[0];
  add_args2.inst_in2 = insts[1];
  add_args2.inst_out = insts[3];

  AddTaskArgs add_args3;
  add_args3.is = is;
  add_args3.inst_in1 = insts[2];
  add_args3.inst_in2 = insts[3];
  add_args3.inst_out = insts[4];

  VerifyTaskArgs verify_args;
  verify_args.is = is;
  verify_args.val = 1.25;
  verify_args.inst = insts[4];

  // first: without subgraph
  {
    std::vector<Event> add_events(TestConfig::num_iters);
    long long t1 = Clock::current_time_in_nanoseconds();
    UserEvent user_event = UserEvent::create_user_event();
    for (int i = 0; i < TestConfig::num_iters; i++) {
      Event pre_cond_event;
      if (i == 0) {
        pre_cond_event = user_event;
      } else {
        pre_cond_event = add_events[i-1];
      }
      Event e1 = p1.spawn(INIT_TASK, &init_args1, sizeof(InitTaskArgs), pre_cond_event);
      Event e2 = p2.spawn(INIT_TASK, &init_args2, sizeof(InitTaskArgs), pre_cond_event);
      Event e12 = Event::merge_events(e1, e2);
      Event e3 = p1.spawn(ADD_TASK, &add_args1, sizeof(AddTaskArgs), e12);
      Event e4 = p2.spawn(ADD_TASK, &add_args2, sizeof(AddTaskArgs), e12);
      Event e34 = Event::merge_events(e3, e4);
      add_events[i] = p1.spawn(ADD_TASK, &add_args3, sizeof(AddTaskArgs), e34);
    }

    long long t2 = Clock::current_time_in_nanoseconds();
    user_event.trigger();
    add_events[TestConfig::num_iters-1].wait();
    long long t3 = Clock::current_time_in_nanoseconds();
    log_app.print("Without subgraph, from constructing graph %.2f us, after constructing graph %.2f us", 
                  (t3 - t1)/1e3, (t3 - t2)/1e3);
    p.spawn(VERIFY_TASK, &verify_args, sizeof(VerifyTaskArgs)).wait();
  }


  // second: with subgraph

  // reset inst
  for (int i = 0; i < 5; i++) {
    double fill_value = 1.25;
    std::vector<CopySrcDstField> sdf(1);
    sdf[0].inst = insts[i];
    sdf[0].field_id = FID_DATA;
    sdf[0].size = sizeof(double);
    is.fill(sdf, ProfilingRequestSet(), &fill_value, sizeof(fill_value)).wait();
    VerifyTaskArgs args;
    args.is = is;
    args.val = 1.25;
    args.inst = insts[i];
    p.spawn(VERIFY_TASK, &args, sizeof(args)).wait();
  }

  // create the subgraph
  long long t0 = Clock::current_time_in_nanoseconds();
  SubgraphDefinition sd;
  sd.tasks.resize(5);
  sd.tasks[0].proc = p1;
  sd.tasks[0].task_id = INIT_TASK;
  sd.tasks[0].args.set(&init_args1, sizeof(InitTaskArgs));
  sd.tasks[1].proc = p2;
  sd.tasks[1].task_id = INIT_TASK;
  sd.tasks[1].args.set(&init_args2, sizeof(InitTaskArgs));
  sd.tasks[2].proc = p1;
  sd.tasks[2].task_id = ADD_TASK;
  sd.tasks[2].args.set(&add_args1, sizeof(AddTaskArgs));
  sd.tasks[3].proc = p2;
  sd.tasks[3].task_id = ADD_TASK;
  sd.tasks[3].args.set(&add_args2, sizeof(AddTaskArgs));
  sd.tasks[4].proc = p1;
  sd.tasks[4].task_id = ADD_TASK;
  sd.tasks[4].args.set(&add_args3, sizeof(AddTaskArgs));

  sd.dependencies.resize(9);
  sd.dependencies[0].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[0].src_op_index = 0;
  sd.dependencies[0].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[0].tgt_op_index = 2;

  sd.dependencies[1].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[1].src_op_index = 1;
  sd.dependencies[1].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[1].tgt_op_index = 2;

  sd.dependencies[2].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[2].src_op_index = 0;
  sd.dependencies[2].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[2].tgt_op_index = 3;

  sd.dependencies[3].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[3].src_op_index = 1;
  sd.dependencies[3].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[3].tgt_op_index = 3;

  sd.dependencies[4].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[4].src_op_index = 2;
  sd.dependencies[4].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[4].tgt_op_index = 4;

  sd.dependencies[5].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[5].src_op_index = 3;
  sd.dependencies[5].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[5].tgt_op_index = 4;

  sd.dependencies[6].src_op_kind = SubgraphDefinition::OPKIND_EXT_PRECOND;
  sd.dependencies[6].src_op_index = 0;
  sd.dependencies[6].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[6].tgt_op_index = 0;

  sd.dependencies[7].src_op_kind = SubgraphDefinition::OPKIND_EXT_PRECOND;
  sd.dependencies[7].src_op_index = 0;
  sd.dependencies[7].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[7].tgt_op_index = 1;

  sd.dependencies[8].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[8].src_op_index = 4;
  sd.dependencies[8].tgt_op_kind = SubgraphDefinition::OPKIND_EXT_POSTCOND;
  sd.dependencies[8].tgt_op_index = 0;

  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();

  // launch the subgraph
  {
    long long t1 = Clock::current_time_in_nanoseconds();
    std::vector<Event> finish_events(TestConfig::num_iters);
    std::vector<Event> postcond_events(TestConfig::num_iters);
    UserEvent user_event = UserEvent::create_user_event();
    for(int i = 0; i < TestConfig::num_iters; i++) {
      std::vector<Event> preconds(1);
      if (i == 0) {
        preconds[0] = user_event;
      } else {
        preconds[0] = postcond_events[i-1];
      }
      std::vector<Event> postconds(1);
      finish_events[i] = sg.instantiate(NULL, 0, ProfilingRequestSet(), preconds, postconds);
      postcond_events[i] = postconds[0];
    }
    Event e = Event::merge_events(finish_events);
    long long t2 = Clock::current_time_in_nanoseconds();
    user_event.trigger();
    postcond_events[TestConfig::num_iters-1].wait();
    e.wait();
    long long t3 = Clock::current_time_in_nanoseconds();
    log_app.print("With subgraph, from creating subgraph %.2f us, from instantiating subgraph %.2f us, after instantiating subgraph %.2f us", 
                  (t3 - t0)/1e3, (t3 - t1)/1e3, (t3 - t2)/1e3);
    p.spawn(VERIFY_TASK, &verify_args, sizeof(VerifyTaskArgs)).wait();
  }

  for (int i = 0; i < 5; i++) {
    insts[i].destroy();
  }
}

int main(int argc, char **argv) 
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  CommandLineParser cp;
  cp.add_option_int("-ni", TestConfig::num_iters);
  cp.add_option_int("-ne", TestConfig::num_elements);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet()).external_wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, ADD_TASK,
                                   CodeDescriptor(add_task),
                                   ProfilingRequestSet()).external_wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, INIT_TASK,
                                   CodeDescriptor(init_task),
                                   ProfilingRequestSet()).external_wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, VERIFY_TASK,
                                   CodeDescriptor(verify_task),
                                   ProfilingRequestSet()).external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);

  rt.shutdown(e);

  int ret = rt.wait_for_shutdown();

  return ret;
}
