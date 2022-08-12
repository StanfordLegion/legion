// test ability to generically create ExternalInstanceResources and use them
//  to create aliases of existing instances

#include "realm.h"
#include "realm/cmdline.h"
#include "realm/deppart/inst_helper.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>

#include "osdep.h"
#include "philox.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  POKE_TASK,
};

enum {
  FID_DATA = 77,
  FID_ALIAS = 88,
};

template <int N>
struct LayoutPermutation {
  int dim_order[N];
  char name[N+1];
};

namespace TestConfig {
  size_t min_size = 8;  // min extent in any dimension
  size_t max_size = 16; // max extent in any dimension
  unsigned dim_mask = ~0U; // i.e. all the dimensions
  unsigned random_seed = 12345;
  bool continue_on_error = false;
};

template <int N, typename T, typename FT>
size_t do_single_dim(Memory src_mem, unsigned seq_no)
{
  // figure out absolute, shrunk (abs and rel) dimensions and
  //  flat dimensions
  Rect<N,T> abs_extent, rel_extent, shrunk_extent;
  size_t elements = 1;

  unsigned ctr = 0;  // for reproducible random values
  for(int i = 0; i < N; i++) {
    T offset = Philox_2x32<>::rand_int(TestConfig::random_seed, seq_no, ctr++,
                                       TestConfig::max_size);
    T extent = (TestConfig::min_size + 1 +
                Philox_2x32<>::rand_int(TestConfig::random_seed, seq_no, ctr++,
                                        (TestConfig::max_size -
                                         TestConfig::min_size)));
    T shrink_lo = Philox_2x32<>::rand_int(TestConfig::random_seed, seq_no, ctr++,
                                          extent + 1);
    T shrink_hi = Philox_2x32<>::rand_int(TestConfig::random_seed, seq_no, ctr++,
                                          extent + 1);
    if(shrink_lo > shrink_hi)
      std::swap(shrink_lo, shrink_hi);

    abs_extent.lo[i] = offset;
    abs_extent.hi[i] = offset + extent;
    shrunk_extent.lo[i] = offset + shrink_lo;
    shrunk_extent.hi[i] = offset + shrink_hi;
    rel_extent.lo[i] = 0;
    rel_extent.hi[i] = shrink_hi - shrink_lo;

    elements *= (extent + 1);
  }

  Rect<1,T> flat_extent(0, elements-1);
  size_t bytes_needed = elements * sizeof(FT);

  // create starting and ending instances to be used for all tests
  RegionInstance start_inst, check_inst;
  InstanceLayoutGeneric *abs_layout;
  InstanceLayoutGeneric *rel_layout;
  {
    std::map<FieldID, size_t> fields;
    fields[FID_DATA] = sizeof(FT);
    InstanceLayoutConstraints ilc(fields, 0 /*block size*/);
    int dim_order[N];
    for(int i = 0; i < N; i++) dim_order[i] = i;
    abs_layout = InstanceLayoutGeneric::choose_instance_layout<N,T>(abs_extent,
                                                                    ilc,
                                                                    dim_order);
  }
  {
    // for the matching rel_layout, we need an extent with the same size
    //  as abs_extent but starting at 0,0, and a different field ID
    std::map<FieldID, size_t> fields;
    fields[FID_ALIAS] = sizeof(FT);
    InstanceLayoutConstraints ilc(fields, 0 /*block size*/);
    Rect<N,T> full_rel = Rect<N,T>(Point<N,T>(0),
                                   abs_extent.hi - abs_extent.lo);
    int dim_order[N];
    for(int i = 0; i < N; i++) dim_order[i] = i;
    rel_layout = InstanceLayoutGeneric::choose_instance_layout<N,T>(full_rel,
                                                                    ilc,
                                                                    dim_order);
  }
  RegionInstance::create_instance(start_inst, src_mem, abs_layout->clone(),
                                  ProfilingRequestSet()).wait();
  RegionInstance::create_instance(check_inst, src_mem, abs_layout->clone(),
                                  ProfilingRequestSet()).wait();

  // create a relatively-indexed alias for the start instance
  RegionInstance start_inst_rel;
  {
    ExternalInstanceResource *extres = start_inst.generate_resource_info(shrunk_extent, FID_DATA, true /*read_only*/);
    if(!extres) {
      log_app.fatal() << "could not generate resource info for instance " << start_inst;
      abort();
    }
    RegionInstance::create_external_instance(start_inst_rel,
                                             extres->suggested_memory(),
                                             rel_layout->clone(),
                                             *extres,
                                             ProfilingRequestSet()).wait();
    delete extres;
  }

  // also test a "flattened" (i.e. 1D) alias the full instance
  RegionInstance start_inst_flat;
  InstanceLayoutGeneric *flat_layout;
  {
    std::map<FieldID, size_t> fields;
    fields[FID_ALIAS] = sizeof(FT);
    InstanceLayoutConstraints ilc(fields, 0 /*block size*/);
    int dim_order[1];
    dim_order[0] = 0;
    flat_layout = InstanceLayoutGeneric::choose_instance_layout<1,T>(flat_extent,
                                                                     ilc,
                                                                     dim_order);

    ExternalInstanceResource *extres = start_inst.generate_resource_info(true /*read_only*/);
    if(!extres) {
      log_app.fatal() << "could not generate subset resource info for instance " << start_inst;
      abort();
    }
    RegionInstance::create_external_instance(start_inst_flat,
                                             extres->suggested_memory(),
                                             flat_layout->clone(),
                                             *extres,
                                             ProfilingRequestSet()).wait();
    delete extres;
  }

  // fill the starting instance with something interesting
  {
    GenericAccessor<FT,N,T> ga(start_inst, FID_DATA);
    T val = 100;
    for(PointInRectIterator<N,T> pir(abs_extent); pir.valid; pir.step()) {
      ga[pir.p] = val;
      val += 1;
    }
  }

  size_t errors = 0;
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.same_address_space_as(src_mem);
  mq.has_capacity(bytes_needed);
  for(Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it) {
    Memory tgt_mem = *it;

    // TODO: get rid of these exceptions by supporting generate_resource_info
    //  in the corresponding Realm memory implementations
    if((tgt_mem.kind() == Memory::Kind::DISK_MEM) ||
       (tgt_mem.kind() == Memory::Kind::HDF_MEM) ||
       (tgt_mem.kind() == Memory::Kind::FILE_MEM) ||
       (tgt_mem.kind() == Memory::Kind::GPU_MANAGED_MEM)) {
      log_app.info() << "skipping " << tgt_mem;
      continue;
    }

    log_app.info() << "testing " << tgt_mem;

    // create two target instances - use the second (first is a dummy to
    //  improve coverage by making sure the instance isn't at the start of the
    //  memory)
    RegionInstance tgt_inst_dummy, tgt_inst_abs;
    RegionInstance::create_instance(tgt_inst_dummy, tgt_mem,
                                    abs_layout->clone(),
                                    ProfilingRequestSet()).wait();
    RegionInstance::create_instance(tgt_inst_abs, tgt_mem,
                                    abs_layout->clone(),
                                    ProfilingRequestSet()).wait();

    // create a relatively-indexed alias for the target instance
    RegionInstance tgt_inst_rel;
    {
      ExternalInstanceResource *extres = tgt_inst_abs.generate_resource_info(shrunk_extent, FID_DATA, false /*!read_only*/);
      if(!extres) {
        log_app.fatal() << "could not generate resource info for instance " << tgt_inst_abs;
        abort();
      }
      RegionInstance::create_external_instance(tgt_inst_rel,
                                               extres->suggested_memory(),
                                               rel_layout->clone(),
                                               *extres,
                                               ProfilingRequestSet()).wait();
      delete extres;
    }

    // fill target instance with 1's
    Event e;
    {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_fill<FT>(1);
      dsts[0].set_field(tgt_inst_abs, FID_DATA, sizeof(FT));
      e = abs_extent.copy(srcs, dsts, ProfilingRequestSet());
    }

    // copy start instance to target instance (via alias!)
    {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(start_inst_rel, FID_ALIAS, sizeof(FT));
      dsts[0].set_field(tgt_inst_rel, FID_ALIAS, sizeof(FT));
      e = rel_extent.copy(srcs, dsts, ProfilingRequestSet(), e);
    }

    // fill check instance so we don't see previous results (NOTE: not
    //  dependent on previous events)
    {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_fill<FT>(2);
      dsts[0].set_field(check_inst, FID_DATA, sizeof(FT));
      Event e2 = abs_extent.copy(srcs, dsts, ProfilingRequestSet());
      e = Event::merge_events(e, e2);
    }

    // now copy entire target instance back to be checked
    {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(tgt_inst_abs, FID_DATA, sizeof(FT));
      dsts[0].set_field(check_inst, FID_DATA, sizeof(FT));
      e = abs_extent.copy(srcs, dsts, ProfilingRequestSet(), e);
    }

    // wait for all the above to actually happen
    e.wait();

    // now check results
    {
      GenericAccessor<FT,N,T> ga(check_inst, FID_DATA);
      T rawval = 100;
      for(PointInRectIterator<N,T> pir(abs_extent); pir.valid; pir.step()) {
        T expval = (shrunk_extent.contains(pir.p) ? rawval : 1);
        T actval = ga[pir.p];
        if(actval == expval) {
          // good
        } else {
          if(!errors)
            log_app.error() << "mismatch: inst=" << tgt_inst_abs << " abs=" << abs_extent << " shrink=" << shrunk_extent << " rel=" << rel_extent;
          log_app.print() << "at " << pir.p << ": exp=" << expval << " act=" << actval;
          errors++;
        }
        rawval += 1;
      }
    }

    tgt_inst_rel.destroy();
    if(errors && !TestConfig::continue_on_error) {
      tgt_inst_abs.destroy();
      tgt_inst_dummy.destroy();
      break;
    }

    // now same routine with the flattened alias
    RegionInstance tgt_inst_flat;
    {
      ExternalInstanceResource *extres = tgt_inst_abs.generate_resource_info(false /*!read_only*/);
      if(!extres) {
        log_app.fatal() << "could not generate subset resource info for instance " << tgt_inst_abs;
        abort();
      }
      RegionInstance::create_external_instance(tgt_inst_flat,
                                               extres->suggested_memory(),
                                               flat_layout->clone(),
                                               *extres,
                                               ProfilingRequestSet()).wait();
      delete extres;
    }

    // fill target instance with 1's
    {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_fill<FT>(1);
      dsts[0].set_field(tgt_inst_abs, FID_DATA, sizeof(FT));
      e = abs_extent.copy(srcs, dsts, ProfilingRequestSet());
    }

    // copy start instance to target instance (via alias!)
    {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(start_inst_flat, FID_ALIAS, sizeof(FT));
      dsts[0].set_field(tgt_inst_flat, FID_ALIAS, sizeof(FT));
      e = flat_extent.copy(srcs, dsts, ProfilingRequestSet(), e);
    }

    // fill check instance so we don't see previous results (NOTE: not
    //  dependent on previous events)
    {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_fill<FT>(2);
      dsts[0].set_field(check_inst, FID_DATA, sizeof(FT));
      Event e2 = abs_extent.copy(srcs, dsts, ProfilingRequestSet());
      e = Event::merge_events(e, e2);
    }

    // now copy entire target instance back to be checked
    {
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].set_field(tgt_inst_abs, FID_DATA, sizeof(FT));
      dsts[0].set_field(check_inst, FID_DATA, sizeof(FT));
      e = abs_extent.copy(srcs, dsts, ProfilingRequestSet(), e);
    }

    // wait for all the above to actually happen
    e.wait();

    // now check results
    {
      GenericAccessor<FT,N,T> ga(check_inst, FID_DATA);
      T expval = 100;
      for(PointInRectIterator<N,T> pir(abs_extent); pir.valid; pir.step()) {
        T actval = ga[pir.p];
        if(actval == expval) {
          // good
        } else {
          if(!errors)
            log_app.error() << "mismatch: inst=" << tgt_inst_abs << " abs=" << abs_extent << " flat=" << flat_extent;
          log_app.print() << "at " << pir.p << ": exp=" << expval << " act=" << actval;
          errors++;
        }
        expval += 1;
      }
    }

    tgt_inst_flat.destroy();
    tgt_inst_abs.destroy();
    tgt_inst_dummy.destroy();
    if(errors && !TestConfig::continue_on_error)
      break;
  }

  start_inst_flat.destroy();
  start_inst_rel.destroy();
  start_inst.destroy();
  check_inst.destroy();
  delete abs_layout;
  delete rel_layout;
  delete flat_layout;

  return errors;
}

std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  // pick a system memory we can see to be our starting memory
  Machine::MemoryQuery mq(Machine::get_machine());
  mq.only_kind(Memory::SYSTEM_MEM);
  mq.has_affinity_to(p);
  Memory src_mem = mq.first();
  assert(src_mem.exists());

  unsigned seq_no = 0;
  size_t errors = 0;
  typedef int FT;

#define DOIT(N,T) \
  { \
    if(((errors == 0) || TestConfig::continue_on_error) && (TestConfig::dim_mask & (1 << (N - 1))) != 0) \
      errors += do_single_dim<N, T, FT>(src_mem, seq_no); \
    seq_no++; /* increment sequence number even if we don't do a test */ \
  }
  FOREACH_NT(DOIT)
#undef DOIT

  Runtime::get_runtime().shutdown(Event::NO_EVENT,
                                  (errors ? 1 : 0));
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-maxsize", TestConfig::max_size)
    .add_option_int("-minsize", TestConfig::min_size)
    .add_option_int("-dims", TestConfig::dim_mask)
    .add_option_int("-seed", TestConfig::random_seed)
    .add_option_bool("-cont", TestConfig::continue_on_error);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

#if 0
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
				   COPYPROF_TASK,
				   CodeDescriptor(copy_profiling_task),
				   ProfilingRequestSet(),
				   0, 0).wait();
#endif

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single task
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // top level task will request shutdown

  // now sleep this thread until that shutdown actually happens
  int result = rt.wait_for_shutdown();
  
  return result;
}
