#include <time.h>

#include <cassert>

#include "realm/cmdline.h"
#include "realm/runtime.h"
#include "realm.h"
#include "realm/id.h"
#include "realm/network.h"
#include "osdep.h"

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  NODE_TASK,
};

constexpr int NUM_INSTS = 2;
constexpr int TEST_CASES = 10;

struct TaskArgs {
  IndexSpace<1> lhs[TEST_CASES][NUM_INSTS];
  IndexSpace<1> rhs[TEST_CASES][NUM_INSTS];
};

namespace TestConfig {
  bool remote_create = false;
};

void node_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  TaskArgs &task_args = *(TaskArgs *)args;

  {
    IndexSpace<1> result;

    std::vector<Rect<1>> rects0;
    rects0.push_back(Rect<1>(Point<1>(0), Point<1>(5)));
    rects0.push_back(Rect<1>(Point<1>(10), Point<1>(15)));

    IndexSpace<1> is0(rects0);

    Event e2 = IndexSpace<1>::compute_union(std::vector<IndexSpace<1>>{is0}, result,
                                            ProfilingRequestSet());
    assert(!result.dense());
    assert(result.sparsity == is0.sparsity);
    result.destroy(e2);
    is0.destroy(e2);
    e2.wait();
  }

  {
    IndexSpace<1> result;

    std::vector<Rect<1>> rects0;
    rects0.push_back(Rect<1>(Point<1>(0), Point<1>(5)));
    rects0.push_back(Rect<1>(Point<1>(10), Point<1>(15)));

    std::vector<Rect<1>> rects1;
    rects1.push_back(Rect<1>(Point<1>(2), Point<1>(8)));

    IndexSpace<1> is0(rects0);
    IndexSpace<1> is1(rects1);

    Event e2 = IndexSpace<1>::compute_union(std::vector<IndexSpace<1>>{is0, is1}, result,
                                            ProfilingRequestSet());
    assert(!result.dense());
    result.destroy(e2);
    is0.destroy(e2);
    e2.wait();
  }

  {
    IndexSpace<1> result;

    std::vector<Rect<1>> rects0;
    rects0.push_back(Rect<1>(Point<1>(0), Point<1>(5)));

    std::vector<Rect<1>> rects1;
    rects1.push_back(Rect<1>(Point<1>(2), Point<1>(8)));

    Event e2 = IndexSpace<1>::compute_union(
        std::vector<IndexSpace<1>>{IndexSpace<1>(rects0), IndexSpace<1>(rects1)}, result,
        ProfilingRequestSet());
    assert(!result.dense());
    result.destroy(e2);
    e2.wait();
  }

  {
    IndexSpace<1> result;

    std::vector<Rect<1>> rects0;
    rects0.push_back(Rect<1>(Point<1>(0), Point<1>(5)));

    std::vector<Rect<1>> rects1;
    rects1.push_back(Rect<1>(Point<1>(2), Point<1>(8)));
    rects1.push_back(Rect<1>(Point<1>(12), Point<1>(14)));

    IndexSpace<1> is0(rects0);
    IndexSpace<1> is1(rects1);
    assert(is1.dense() == false);

    Event e2 = IndexSpace<1>::compute_intersection(std::vector<IndexSpace<1>>{is0, is1},
                                                   result, ProfilingRequestSet());
    assert(result.sparsity == is1.sparsity);
    result.destroy(e2);
    is1.destroy(e2);
    e2.wait();
  }

  {
    IndexSpace<1> result;

    std::vector<Rect<1>> rects1;
    rects1.push_back(Rect<1>(Point<1>(2), Point<1>(8)));
    rects1.push_back(Rect<1>(Point<1>(12), Point<1>(14)));

    IndexSpace<1> is1(rects1);

    Event e2 = IndexSpace<1>::compute_intersection(std::vector<IndexSpace<1>>{is1},
                                                   result, ProfilingRequestSet());

    assert(result.sparsity.exists());
    assert(result.sparsity == is1.sparsity);

    result.destroy(e2);
    is1.destroy(e2);
    e2.wait();
  }

  {
    IndexSpace<1> result;

    std::vector<Rect<1>> rects0;
    rects0.push_back(Rect<1>(Point<1>(0), Point<1>(5)));
    rects0.push_back(Rect<1>(Point<1>(10), Point<1>(15)));

    std::vector<Rect<1>> rects1;
    rects1.push_back(Rect<1>(Point<1>(2), Point<1>(8)));
    rects1.push_back(Rect<1>(Point<1>(12), Point<1>(14)));

    IndexSpace<1> is0(rects0);
    IndexSpace<1> is1(rects1);

    Event e2 = IndexSpace<1>::compute_intersection(std::vector<IndexSpace<1>>{is0, is1},
                                                   result, ProfilingRequestSet());

    assert(result.sparsity != is0.sparsity);
    assert(result.sparsity != is1.sparsity);
    assert(result.sparsity.exists());

    result.destroy(e2);
    is0.destroy(e2);
    is1.destroy(e2);
    e2.wait();
  }

  {
    // case 3
    IndexSpace<1> result;

    std::vector<Rect<1>> rects0;
    rects0.push_back(Rect<1>(Point<1>(0), Point<1>(5)));

    std::vector<Rect<1>> rects1;
    rects1.push_back(Rect<1>(Point<1>(2), Point<1>(8)));

    Event e2 = IndexSpace<1>::compute_intersection(
        std::vector<IndexSpace<1>>{IndexSpace<1>(rects0), IndexSpace<1>(rects1)}, result,
        ProfilingRequestSet());
    assert(result.dense());
    e2.wait();
  }

  // no overlap
  {
    IndexSpace<1> result;

    std::vector<Rect<1>> rects0;
    rects0.push_back(Rect<1>(Point<1>(0), Point<1>(5)));

    std::vector<Rect<1>> rects1;
    rects1.push_back(Rect<1>(Point<1>(6), Point<1>(8)));
    rects1.push_back(Rect<1>(Point<1>(12), Point<1>(14)));

    IndexSpace<1> is0(rects1);
    IndexSpace<1> is1(rects0);

    std::vector<IndexSpace<1>> results;
    Event e2 = IndexSpace<1>::compute_differences(std::vector<IndexSpace<1>>{is0},
                                                  std::vector<IndexSpace<1>>{is1},
                                                  results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      results[i].destroy(e2);
    }
    is0.destroy(e2);
    e2.wait();
  }

  {
    IndexSpace<1> result;

    std::vector<Rect<1>> rects0;
    rects0.push_back(Rect<1>(Point<1>(0), Point<1>(20)));

    std::vector<Rect<1>> rects1;
    rects1.push_back(Rect<1>(Point<1>(2), Point<1>(4)));
    rects1.push_back(Rect<1>(Point<1>(12), Point<1>(14)));

    IndexSpace<1> is0(rects0);
    IndexSpace<1> is1(rects1);

    Event e2 = IndexSpace<1>::compute_union(std::vector<IndexSpace<1>>{is0, is1}, result,
                                            ProfilingRequestSet());
    assert(result.dense());
    is1.destroy(e2);
    e2.wait();
  }

  int case_id = 0;

  {
    std::vector<IndexSpace<1>> results;
    Event e2 = IndexSpace<1>::compute_differences(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        std::vector<IndexSpace<1>>{std::begin(task_args.rhs[case_id]),
                                   std::end(task_args.rhs[case_id])},
        results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      if(results[i].sparsity.exists()) {
        results[i].destroy(e2);
      }
    }
    e2.wait();
  }

  case_id++;

  {
    std::vector<IndexSpace<1>> results;
    std::vector<IndexSpace<1>> rhs(1, Rect<1>::make_empty());
    Event e2 = IndexSpace<1>::compute_differences(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        rhs, results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      results[i].destroy(e2);
    }
    e2.wait();
  }

  case_id++;

  {
    std::vector<IndexSpace<1>> results;
    Event e2 = IndexSpace<1>::compute_differences(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      results[i].destroy(e2);
    }
    e2.wait();
  }

  case_id++;

  {
    std::vector<IndexSpace<1>> results;
    Event e2 = IndexSpace<1>::compute_unions(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      results[i].destroy(e2);
    }
    e2.wait();
  }

  /*case_id++;

  // empty rhs

  {
    std::vector<IndexSpace<1>> results;
    std::vector<IndexSpace<1>> rhs(1, Rect<1>::make_empty());
    Event e2 = IndexSpace<1>::compute_unions(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        rhs, results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      if(results[i].sparsity.exists()) {
        results[i].destroy(e2);
      }
    }
    e2.wait();
  }*/

  case_id++;

  // TODO: test empty rhs

  {
    std::vector<IndexSpace<1>> results;
    Event e2 = IndexSpace<1>::compute_unions(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        std::vector<IndexSpace<1>>{std::begin(task_args.rhs[case_id]),
                                   std::end(task_args.rhs[case_id])},
        results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      if(results[i].sparsity.exists()) {
        results[i].destroy(e2);
      }
    }
    e2.wait();
  }

  case_id++;

  // empty lhs
  {
    std::vector<IndexSpace<1>> results;
    std::vector<IndexSpace<1>> lhs(1, Rect<1>::make_empty());
    Event e2 = IndexSpace<1>::compute_unions(
        lhs,
        std::vector<IndexSpace<1>>{std::begin(task_args.rhs[case_id]),
                                   std::end(task_args.rhs[case_id])},
        results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      if(results[i].sparsity.exists()) {
        results[i].destroy(e2);
      }
    }
    e2.wait();
  }

  case_id++;

  // int case_id = 0;

  {
    std::vector<IndexSpace<1>> results;
    Event e2 = IndexSpace<1>::compute_intersections(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        std::vector<IndexSpace<1>>{std::begin(task_args.rhs[case_id]),
                                   std::end(task_args.rhs[case_id])},
        results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      results[i].destroy(e2);
    }
    e2.wait();
  }

  case_id++;

  {
    std::vector<IndexSpace<1>> results;
    Event e2 = IndexSpace<1>::compute_intersections(
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        std::vector<IndexSpace<1>>{std::begin(task_args.lhs[case_id]),
                                   std::end(task_args.lhs[case_id])},
        results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      results[i].destroy(e2);
    }
    e2.wait();
  }

  case_id++;

  // dense empty lhs
  {
    std::vector<IndexSpace<1>> results;
    std::vector<IndexSpace<1>> lhs{Rect<1>(4, 6)};
    Event e2 = IndexSpace<1>::compute_intersections(
        lhs,
        std::vector<IndexSpace<1>>{std::begin(task_args.rhs[case_id]),
                                   std::end(task_args.rhs[case_id])},
        results, ProfilingRequestSet());
    for(size_t i = 0; i < results.size(); i++) {
      if(results[i].sparsity.exists()) {
        results[i].destroy(e2);
      }
    }
    e2.wait();
  }
}

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  std::vector<Rect<1>> rects;
  rects.push_back(Rect<1>(Point<1>(0), Point<1>(5)));
  rects.push_back(Rect<1>(Point<1>(10), Point<1>(15)));

  std::vector<Rect<1>> rects1;
  rects1.push_back(Rect<1>(Point<1>(2), Point<1>(4)));
  rects1.push_back(Rect<1>(Point<1>(12), Point<1>(14)));

  Machine machine = Machine::get_machine();
  std::vector<Memory> memories;
  for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it;
      ++it) {
    Memory m = *it;
    if(m.kind() == Memory::SYSTEM_MEM) {
      memories.push_back(m);
    }
  }

  std::vector<Event> events;

  for(std::vector<Memory>::const_iterator it = memories.begin(); it != memories.end();
      ++it) {
    Processor proc = *Machine::ProcessorQuery(machine)
                          .only_kind(Processor::LOC_PROC)
                          .same_address_space_as(*it)
                          .begin();

    std::vector<IndexSpace<1>> roots;
    TaskArgs args;
    for(int id = 0; id < TEST_CASES; id++) {

      roots.push_back(IndexSpace<1>(rects));
      //roots.back().sparsity.add_references();

      std::vector<IndexSpace<1>> lhs;
      roots.back()
          .create_equal_subspaces(NUM_INSTS, 1, lhs, Realm::ProfilingRequestSet())
          .wait();

      roots.push_back(IndexSpace<1>(rects1));
      //roots.back().sparsity.add_references();

      std::vector<IndexSpace<1>> rhs_diff;
      roots.back()
          .create_equal_subspaces(NUM_INSTS, 1, rhs_diff, Realm::ProfilingRequestSet())
          .wait();

      for(int i = 0; i < NUM_INSTS; i++) {
        args.lhs[id][i] = lhs[i];
        args.rhs[id][i] = rhs_diff[i];
      }
    }

    /*if((TestConfig::remote_create) &&
       NodeID(ID(*it).memory_owner_node()) ==
           NodeID(ID(ptr_data[0].inst).instance_owner_node())) {
      continue;
    }*/

    proc.spawn(NODE_TASK, &args, sizeof(args)).wait();

    for(size_t i = 0; i < roots.size(); i++) {
      assert(roots[i].sparsity.exists());
      roots[i].destroy();
    }
  }

  usleep(100000);
  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(), 0);
}

int main(int argc, char **argv)
{
  Runtime rt;

  CommandLineParser cp;
  cp.add_option_int_units("-remote_create", TestConfig::remote_create);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.init(&argc, &argv);
  rt.register_task(MAIN_TASK, main_task);
  Processor::register_task_by_kind(Processor::LOC_PROC, /*!global=*/false, NODE_TASK,
                                   CodeDescriptor(node_task), ProfilingRequestSet(), 0, 0)
      .wait();

  ModuleConfig *core = Runtime::get_runtime().get_module_config("core");
  assert(core->set_property("enable_sparsity_refcount", 1));

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  rt.collective_spawn(p, MAIN_TASK, 0, 0);
  int ret = rt.wait_for_shutdown();
  return ret;
}
