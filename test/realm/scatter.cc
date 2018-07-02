#include "realm.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>
#include <unistd.h>

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
};

enum {
  FID_PTR1 = 100,
  FID_DATA1,
  FID_DATA2,
};

struct SpeedTestArgs {
  Memory mem;
  RegionInstance inst;
  size_t elements;
  int reps;
  Machine::AffinityDetails affinity;
};

template <int N, typename T, typename DT>
void dump_field(RegionInstance inst, FieldID fid, IndexSpace<N,T> is)
{
  AffineAccessor<DT, N, T> acc(inst, fid);
  for(IndexSpaceIterator<N,T> it(is); it.valid; it.step())
    for(PointInRectIterator<N,T> it2(it.rect); it2.valid; it2.step()) {
      DT v = acc[it2.p];
      std::cout << it2.p << ": " << v << "\n";
    }
}

template <int N, typename T, int N2, typename T2, typename DT>
bool scatter_gather_test(Memory m, T size1, T2 size2)
{
  Rect<N,T> r1;
  Rect<N2,T2> r2;
  for(int i = 0; i < N; i++) r1.lo[i] = 0;
  for(int i = 0; i < N; i++) r1.hi[i] = size1 - 1;
  for(int i = 0; i < N2; i++) r2.lo[i] = 0;
  for(int i = 0; i < N2; i++) r2.hi[i] = size2 - 1;
  IndexSpace<N,T> is1(r1);
  IndexSpace<N2,T2> is2(r2);

  RegionInstance inst1, inst2a, inst2b;

  std::map<FieldID, size_t> fields1;
  fields1[FID_PTR1] = sizeof(Point<N2,T2>);
  fields1[FID_DATA1] = sizeof(DT);
  fields1[FID_DATA2] = sizeof(DT);
  RegionInstance::create_instance(inst1, m, is1, fields1,
				  0 /*SOA*/, ProfilingRequestSet()).wait();

  std::map<FieldID, size_t> fields2;
  fields2[FID_DATA1] = sizeof(DT);
  fields2[FID_DATA2] = sizeof(DT);
  RegionInstance::create_instance(inst2a, m, is2, fields2,
				  0 /*SOA*/, ProfilingRequestSet()).wait();
  RegionInstance::create_instance(inst2b, m, is2, fields2,
				  0 /*SOA*/, ProfilingRequestSet()).wait();

  // fill the new instance
  {
    AffineAccessor<Point<N2, T2>, N, T> acc_ptr1(inst1, FID_PTR1);
    AffineAccessor<DT, N, T> acc_data1(inst1, FID_DATA1);

    DT count = 0;
    IndexSpaceIterator<N, T> it1(is1);
    IndexSpaceIterator<N2, T2> it2(is2);
    assert(it1.valid);
    assert(it2.valid);
    PointInRectIterator<N, T> pit1(it1.rect);
    PointInRectIterator<N2, T2> pit2(it2.rect);
    while(true) {
      acc_ptr1[pit1.p] = pit2.p;
      acc_data1[pit1.p] = count;
      count += 1;
      if(!pit1.step()) {
	if(!it1.step()) break;
	pit1.reset(it1.rect);
      }
      if(!pit2.step()) {
	if(!it2.step()) it2.reset(is2);
	pit2.reset(it2.rect);
      }
    }
  }

  dump_field<N, T, Point<N2, T2> >(inst1, FID_PTR1, is1);
  dump_field<N, T, DT >(inst1, FID_DATA1, is1);

  {
    // affine copy within inst1
    Matrix<N, N, T> xform;
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++)
	xform.rows[i][j] = (i == j) ? -1 : 0;
    Point<N, T> offset(r1.hi);
    typename CopyIndirection<N,T>::template Affine<N,T> indirect;
    indirect.transform = xform;
    indirect.offset_lo = offset;
    indirect.offset_hi = offset;
    for(int i = 0; i < N; i++) indirect.divisor[i] = 1;
    indirect.spaces.push_back(is2);
    indirect.insts.push_back(inst2a);

    std::vector<CopySrcDstField> srcs, dsts;
    srcs.resize(1);
    dsts.resize(1);
#ifdef ACTUALLY_TEST_GATHER
    srcs[0].set_indirect(0, FID_DATA1, sizeof(DT));
#else
    srcs[0].set_field(inst1, FID_DATA1, sizeof(DT));
#endif
    //srcs[0].template set_fill<DT>(2.5);
    dsts[0].set_field(inst1, FID_DATA2, sizeof(DT));

    is1.copy(srcs, dsts, 
	     std::vector<const typename CopyIndirection<N,T>::Base *>(1, &indirect),
	     ProfilingRequestSet()).wait();
  }

  dump_field<N, T, DT >(inst1, FID_DATA2, is1);

  inst1.destroy();
  inst2a.destroy();
  inst2b.destroy();

  return true;
}

std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Realm scatter/gather test";

  // build the list of memories that we want to test
  Memory m = Machine::MemoryQuery(Machine::get_machine()).only_kind(Memory::SYSTEM_MEM).first();
  assert(m.exists());

  scatter_gather_test<1, int, 1, int, float>(m, 10, 8);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

#if 0
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-b")) {
      buffer_size = strtoll(argv[++i], 0, 10);
      continue;
    }

  }
#endif

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();
  
  return 0;
}
