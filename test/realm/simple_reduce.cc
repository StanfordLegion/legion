#include "realm.h"
#include "realm/id.h"
#include "realm/cmdline.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  MEMSPEED_TASK,
  COPYPROF_TASK,
};

enum {
  FID_INT = 44,
  FID_DOUBLE = 88,
};

enum {
  REDOP_MIXED_ADD = 99,
};

class ReductionOpMixedAdd {
public:
  typedef double LHS;
  typedef int RHS;

  template <bool EXCL>
  static void apply(LHS& lhs, RHS rhs) { lhs += rhs; }

  // both of these are optional
  static const RHS identity;

  template <bool EXCL>
  static void fold(RHS& rhs1, RHS rhs2) { rhs1 += rhs2; }
};

const ReductionOpMixedAdd::RHS ReductionOpMixedAdd::identity = 0;

template <int N, typename T>
bool test_reduction(IndexSpace<N,T> domain, IndexSpace<N,T> bloat,
                    const std::vector<Memory>& mems)
{
  std::map<FieldID, size_t> fields;
  fields[FID_INT] = sizeof(int);
  fields[FID_DOUBLE] = sizeof(double);

  RegionInstance dst_inst;
  std::vector<RegionInstance> src_insts;

  RegionInstance::create_instance(dst_inst, mems[0], bloat,
                                  fields, 0 /*SOA*/,
                                  ProfilingRequestSet()).wait();

  src_insts.resize(mems.size());
  for(size_t i = 0; i < mems.size(); i++)
    RegionInstance::create_instance(src_insts[i], mems[i], bloat,
                                    fields, 0 /*SOA*/,
                                    ProfilingRequestSet()).wait();

  std::vector<Event> events;
  {
    {
      std::vector<CopySrcDstField> srcs(2), dsts(2);
      srcs[0].set_fill<int>(1);
      srcs[1].set_fill<double>(2);
      dsts[0].set_field(dst_inst, FID_INT, sizeof(int));
      dsts[1].set_field(dst_inst, FID_DOUBLE, sizeof(double));
      events.push_back(bloat.copy(srcs, dsts, ProfilingRequestSet()));
    }

    for(size_t i = 0; i < mems.size(); i++) {
      std::vector<CopySrcDstField> srcs(2), dsts(2);
      srcs[0].set_fill<int>(2*i + 3);
      srcs[1].set_fill<double>(2*i + 4);
      dsts[0].set_field(src_insts[i], FID_INT, sizeof(int));
      dsts[1].set_field(src_insts[i], FID_DOUBLE, sizeof(double));
      events.push_back(bloat.copy(srcs, dsts, ProfilingRequestSet()));
    }
  }
  Event::merge_events(events).wait();
  events.clear();

  size_t errors = 0;
  // test direct application first
  {
    Event e = Event::NO_EVENT;
    for(size_t i = 0; i < mems.size(); i++) {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(src_insts[i], FID_INT, sizeof(int));
      dsts[0].set_field(dst_inst, FID_DOUBLE, sizeof(double));
      dsts[0].set_redop(REDOP_MIXED_ADD, false /*!is_fold*/);
      e = domain.copy(srcs, dsts, ProfilingRequestSet(), e);
    }
    e.wait();

    AffineAccessor<double,N,T> acc(dst_inst, FID_DOUBLE);
    for(IndexSpaceIterator<N,T> it(domain); it.valid; it.step())
      for(PointInRectIterator<N,T> it2(it.rect); it2.valid; it2.step()) {
        double exp = 1 + (mems.size() + 1) * (mems.size() + 1);
        double act = acc[it2.p];
        if(act == exp) {
          // good
        } else {
          if(++errors < 10)
            log_app.error() << "apply mismatch: [" << it2.p << "] = " << act << " (expected " << exp << ")";
        }
      }
  }

  // test direct application first
  {
    Event e = Event::NO_EVENT;
    for(size_t i = 0; i < mems.size(); i++) {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(src_insts[i], FID_INT, sizeof(int));
      dsts[0].set_field(dst_inst, FID_INT, sizeof(int));
      dsts[0].set_redop(REDOP_MIXED_ADD, true /*is_fold*/);
      e = domain.copy(srcs, dsts, ProfilingRequestSet(), e);
    }
    e.wait();

    AffineAccessor<int,N,T> acc(dst_inst, FID_INT);
    for(IndexSpaceIterator<N,T> it(domain); it.valid; it.step())
      for(PointInRectIterator<N,T> it2(it.rect); it2.valid; it2.step()) {
        int exp = (mems.size() + 1) * (mems.size() + 1);
        int act = acc[it2.p];
        if(act != exp) {
          if(++errors < 10)
            log_app.error() << "fold mismatch: [" << it2.p << "] = " << act << " (expected " << exp << ")";
        }
      }
  }

  dst_inst.destroy();
  for(size_t i = 0; i < src_insts.size(); i++)
    src_insts[i].destroy();

  return (errors == 0);
}

void top_level_task(const void *data, size_t datalen,
                    const void *userdata, size_t userlen, Processor p)
{
  std::vector<Memory> mems;
  Machine::MemoryQuery mq(Machine::get_machine());
  for(Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it)
    if((*it).capacity() > 0)
      mems.push_back(*it);

  bool ok = true;

  {
    // 1-D
    Rect<1> r(0,7);
    Rect<1> smaller(0,7);
  
    if(ok) ok = test_reduction(IndexSpace<1>(r), IndexSpace<1>(r), mems);
    if(ok) ok = test_reduction(IndexSpace<1>(smaller), IndexSpace<1>(r), mems);
  }

  {
    // 2-D
    Rect<2> r(Point<2>(1, 8), Point<2>(5, 13));
    Rect<2> smaller(Point<2>(2, 9), Point<2>(4, 12));
    Rect<2> xstrip(Point<2>(2, 9), Point<2>(5, 9));
    Rect<2> ystrip(Point<2>(2, 9), Point<2>(2, 12));
  
    if(ok) ok = test_reduction(IndexSpace<2>(r), IndexSpace<2>(r), mems);
    if(ok) ok = test_reduction(IndexSpace<2>(smaller), IndexSpace<2>(r), mems);
    if(ok) ok = test_reduction(IndexSpace<2>(xstrip), IndexSpace<2>(r), mems);
    if(ok) ok = test_reduction(IndexSpace<2>(ystrip), IndexSpace<2>(r), mems);
  }

  {
    // 3-D
    Rect<3> r(Point<3>(1, 8, 14), Point<3>(5, 13, 17));
    Rect<3> smaller(Point<3>(2, 9, 15), Point<3>(4, 12, 17));
  
    if(ok) ok = test_reduction(IndexSpace<3>(r), IndexSpace<3>(r), mems);
    if(ok) ok = test_reduction(IndexSpace<3>(smaller), IndexSpace<3>(r), mems);
  }

  Runtime::get_runtime().shutdown(Event::NO_EVENT,
				  ok ? 0 : 1);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

#if 0
  CommandLineParser cp;
  cp.add_option_int_units("-b", TestConfig::buffer_size, 'M')
    .add_option_int("-tasks", TestConfig::do_tasks)
    .add_option_int("-copies", TestConfig::do_copies)
    .add_option_int("-reps", TestConfig::copy_reps)
    .add_option_int("-slowmem", TestConfig::slow_mems);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);
#endif

  rt.register_reduction(REDOP_MIXED_ADD,
			ReductionOpUntyped::create_reduction_op<ReductionOpMixedAdd>());


  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single task - top level will issue shutdown
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // now sleep this thread until that shutdown actually happens
  int result = rt.wait_for_shutdown();
  
  return result;
}
