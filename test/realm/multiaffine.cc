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

#include "philox.h"

#include "multiaffine.h"

typedef Philox_2x32<> PRNG;

using namespace Realm;

Logger log_app("app");

namespace TestConfig {
  unsigned dim_mask = 7; // i.e. 1-D, 2-D, 3-D
  unsigned log2_size = 4;
  int random_tests = 20;
  int random_seed = 12345;
};

template <int N, typename T>
void ptr_write_task_cpu(const void *args, size_t arglen,
			const void *userdata, size_t userlen, Processor p)
{
  const PtrWriteTaskArgs<N,T>& targs = *static_cast<const PtrWriteTaskArgs<N,T> *>(args);

  MultiAffineAccessor<void *,N,T> acc(targs.inst, FID_ADDR);

  for(IndexSpaceIterator<N, T> it(targs.space); it.valid; it.step()) {
    for(PointInRectIterator<N,T> pit(it.rect); pit.valid; pit.step()) {
      void **ptr = acc.ptr(pit.p);
      log_app.debug() << "write " << pit.p << ": " << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec;
      *ptr = ptr;
    }
  }
}

template <int N, typename T>
bool test_case(Processor curr_proc, Processor write_proc,
	       IndexSpace<N,T> space, int num_pieces, int seed, int test_id)
{
  // choose memories for instances
  Memory m_check = Machine::MemoryQuery(Machine::get_machine()).best_affinity_to(curr_proc).first();
  Memory m_write = Machine::MemoryQuery(Machine::get_machine()).best_affinity_to(write_proc).first();
  assert(m_check.exists() && m_write.exists());

  // check instance is simple
  RegionInstance inst_check;
  {
    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_ADDR] = sizeof(void *);
    int dim_order[N];
    for(int i = 0; i < N; i++) dim_order[i] = i;
    InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout<N,T>(space,
										    InstanceLayoutConstraints(field_sizes, 0 /*SOA*/),
										    dim_order);
    RegionInstance::create_instance(inst_check, m_check, ilg,
				    ProfilingRequestSet()).wait();
  }

  // write instance is chopped up into pieces and reordered
  int seq_no = 0;
  std::vector<Rect<N,T> > pieces;
  pieces.reserve(num_pieces);
  pieces.push_back(space.bounds);
  for(int i = 1; i < num_pieces; i++) {
    int to_split, split_dim, split_idx;
    do {
      to_split = PRNG::rand_int(seed, test_id, seq_no++, i);
    } while(pieces[to_split].volume() == 1);
    do {
      split_dim = PRNG::rand_int(seed, test_id, seq_no++, N);
    } while(pieces[to_split].lo[split_dim] == pieces[to_split].hi[split_dim]);
    split_idx = (pieces[to_split].lo[split_dim] +
		 PRNG::rand_int(seed, test_id, seq_no++,
				/* no +1 here - we do not want to pick the
				   last value in the range */
				(pieces[to_split].hi[split_dim] -
				 pieces[to_split].lo[split_dim])));
    Rect<N,T> r = pieces[to_split];
    r.hi[split_dim] = split_idx;
    pieces[to_split].lo[split_dim] = split_idx + 1;
    pieces.push_back(r);
  }

  RegionInstance inst_write;
  {
    std::map<FieldID, size_t> field_sizes;
    field_sizes[FID_ADDR] = sizeof(void *);
    int dim_order[N];
    for(int i = 0; i < N; i++) dim_order[i] = i;
    InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout<N,T>(space,
										    pieces,
										    InstanceLayoutConstraints(field_sizes, 0 /*SOA*/),
										    dim_order);
    RegionInstance::create_instance(inst_write, m_write, ilg,
				    ProfilingRequestSet()).wait();
  }

  // zero out write instance
  Event e;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_fill<void *>(0);
    dsts[0].set_field(inst_write, FID_ADDR, sizeof(void *));
    e = space.copy(srcs, dsts, ProfilingRequestSet());
  }

  // run task to write addresses
  {
    PtrWriteTaskArgs<N,T> targs;
    targs.space = space;
    targs.inst = inst_write;
    e = write_proc.spawn(PTR_WRITE_TASK_BASE+N, &targs, sizeof(targs), e);
  }

  // copy data to locally-readable check instance
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    srcs[0].set_field(inst_write, FID_ADDR, sizeof(void *));
    dsts[0].set_field(inst_check, FID_ADDR, sizeof(void *));
    e = space.copy(srcs, dsts, ProfilingRequestSet(), e);
  }
  e.wait();

  // loop over the pieces and make sure they are laid out in the order
  //  expected
  size_t errors = 0;
  {
    uintptr_t base = 0;
    AffineAccessor<void *,N,T> acc(inst_check, FID_ADDR);
    for(int i = 0; i < num_pieces; i++) {
      size_t lpos = 0;
      for(PointInRectIterator<N,T> pit(pieces[i]); pit.valid; pit.step()) {
	lpos++;
	if(!space.contains(pit.p)) continue;

	uintptr_t val = reinterpret_cast<uintptr_t>(acc[pit.p]);
	log_app.debug() << "at " << pit.p << ": " << std::hex << val << std::dec;
	uintptr_t exp;
	if(base == 0) {
	  // everything should be relative to the first point we check
	  base = val - (lpos * sizeof(void *));
	  exp = val;
	} else {
	  exp = base + (lpos * sizeof(void *));
	}
	if(val != exp) {
	  log_app.error() << "at " << pit.p
			  << ": expected " << std::hex << exp
			  << ", actual " << val << std::dec;
	  errors++;
	}
      }
      base += pieces[i].volume() * sizeof(void *);
    }
  }

  inst_check.destroy();
  inst_write.destroy();

  return (errors == 0);
}

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Realm multi-affine instance test";

  // decide which processor we'll do writes on - use a GPU if available
  Processor proc_write = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::TOC_PROC)
    .first();
  if(!proc_write.exists())
    proc_write = p;

  int test_id = 0;
  int errors = 0;
  if((TestConfig::dim_mask & 1) != 0) {
    Rect<1> bounds;
    bounds.lo.x = 0;
    bounds.hi.x = (1 << TestConfig::log2_size) - 1;
    if(!test_case(p, proc_write, IndexSpace<1>(bounds),
		  8, TestConfig::random_seed, test_id))
      errors++;
  }

  if((TestConfig::dim_mask & 2) != 0) {
    int lx2 = (TestConfig::log2_size / 2);
    int ly2 = (TestConfig::log2_size - lx2);
    Rect<2> bounds;
    bounds.lo.x = 0;
    bounds.hi.x = (1 << lx2) - 1;
    bounds.lo.y = 0;
    bounds.hi.y = (1 << ly2) - 1;
    if(!test_case(p, proc_write, IndexSpace<2>(bounds),
		  8, TestConfig::random_seed, test_id))
      errors++;
  }

  // HACK: there's a shutdown race condition related to instance destruction
  usleep(100000);

  Runtime::get_runtime().shutdown(Event::NO_EVENT, errors == 0 ? 0 : 1);
}

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
// defined in multiaffine_gpu.cu
void register_multiaffine_gpu_tasks();
#endif

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int_units("-size", TestConfig::log2_size)
    .add_option_int("-dims", TestConfig::dim_mask)
    .add_option_int("-seed", TestConfig::random_seed);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
				   PTR_WRITE_TASK_BASE + 1,
				   CodeDescriptor(ptr_write_task_cpu<1,int>),
				   ProfilingRequestSet(),
				   0, 0).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
				   PTR_WRITE_TASK_BASE + 2,
				   CodeDescriptor(ptr_write_task_cpu<2,int>),
				   ProfilingRequestSet(),
				   0, 0).wait();

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
  register_multiaffine_gpu_tasks();
#endif

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // now sleep this thread until that shutdown actually happens
  int ret = rt.wait_for_shutdown();
  
  return ret;
}
