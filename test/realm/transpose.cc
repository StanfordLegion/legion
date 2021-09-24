#include "realm.h"
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

template <int N>
struct LayoutPermutation {
  int dim_order[N];
  char name[N+1];
};

template <int N>
struct TransposeExperiment {
  const struct LayoutPermutation<N> *src_perm;
  const struct LayoutPermutation<N> *dst_perm;
  long long nanoseconds;
};

template <int N>
void create_permutations(std::vector<LayoutPermutation<N> >& perms,
			 LayoutPermutation<N>& scratch,
			 int pos, size_t narrow_size)
{
  static const char *dim_names = "XYZW";

  for(int i = 0; i < N; i++) {
    bool found = false;
    for(int j = 0; j < pos; j++)
      if(scratch.dim_order[j] == i) found = true;
    if(found) continue;
    
    scratch.dim_order[pos] = i;
    scratch.name[pos] = (((i == 0) && (narrow_size > 0)) ?
                           tolower(dim_names[i]):
                           dim_names[i]);

    if(pos == (N - 1))
      perms.push_back(scratch);
    else
      create_permutations<N>(perms, scratch, pos+1, narrow_size);
  }
}

struct CopyProfResult {
  long long *nanoseconds;
  UserEvent done;
};

void copy_profiling_task(const void *args, size_t arglen, 
			 const void *userdata, size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(CopyProfResult));
  const CopyProfResult *result = static_cast<const CopyProfResult *>(resp.user_data());

  ProfilingMeasurements::OperationTimeline timeline;
  if(resp.get_measurement(timeline)) {
    *(result->nanoseconds) = timeline.complete_time - timeline.start_time;
    result->done.trigger();
  } else {
    log_app.fatal() << "no operation timeline in profiling response!";
    assert(0);
  }
}

namespace TestConfig {
  size_t buffer_size = 64 << 20; // should be bigger than any cache in system
  size_t narrow_dim = 0;  // should one dimension be a fixed (narrow) size?
  unsigned dim_mask = 7; // i.e. 1-D, 2-D, 3-D
  bool all_mems = false;
};

template <int N, typename FT>
void do_single_dim(Memory src_mem, Memory dst_mem, int log2_size,
                   size_t narrow_size,
		   Processor prof_proc, int pad = 0)
{
  std::vector<LayoutPermutation<N> > perms;
  LayoutPermutation<N> scratch;
  memset(&scratch, 0, sizeof(scratch));
  create_permutations<N>(perms, scratch, 0, narrow_size);

  Rect<N> bounds;
  for(int i = 0; i < N; i++) {
    bounds.lo[i] = 0;
    if((N > 1) && (narrow_size > 0)) {
      if(i == 0)
        bounds.hi[i] = narrow_size - 1;
      else {
        // max is for compilers that don't see this code is unreachable for N==1
        bounds.hi[i] = (1 << (log2_size / std::max(1, N - 1))) - 1;
      }
    } else
      bounds.hi[i] = (1 << (log2_size / N)) - 1;
  }
  IndexSpace<N> is(bounds);

  Rect<N> bounds_pad;
  for(int i = 0; i < N; i++) {
    bounds_pad.lo[i] = 0;
    bounds_pad.hi[i] = bounds.hi[i] + (((narrow_size > 0) && (narrow_size < 32)) ? narrow_size : 32);
  }
  IndexSpace<N> is_pad(bounds_pad);

  std::map<FieldID, size_t> field_sizes;
  field_sizes[0] = sizeof(FT);
  InstanceLayoutConstraints ilc(field_sizes, 1);

  std::vector<TransposeExperiment<N> *> experiments;

  Event wait_for = Event::NO_EVENT;
  std::vector<Event> done_events;
  for(unsigned i = 0; i < perms.size(); i++) {
    RegionInstance src_inst;
    {
      InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout<N,FT>(is_pad, ilc, perms[i].dim_order);
      wait_for = RegionInstance::create_instance(src_inst, src_mem, ilg,
                                                 ProfilingRequestSet(),
                                                 wait_for);
      std::vector<CopySrcDstField> tgt(1);
      tgt[0].inst = src_inst;
      tgt[0].field_id = 0;
      tgt[0].size = field_sizes[0];
      FT fill_value = 77;
      wait_for = is.fill(tgt, ProfilingRequestSet(),
                         &fill_value, sizeof(fill_value), wait_for);
    }

    for(unsigned j = 0; j < perms.size(); j++) {
      RegionInstance dst_inst;
      {
        InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout<N,FT>(is_pad, ilc, perms[j].dim_order);
        wait_for = RegionInstance::create_instance(dst_inst, dst_mem, ilg,
                                                   ProfilingRequestSet(),
                                                   wait_for);
        std::vector<CopySrcDstField> tgt(1);
        tgt[0].inst = dst_inst;
        tgt[0].field_id = 0;
        tgt[0].size = field_sizes[0];
        FT fill_value = 88;
        wait_for = is.fill(tgt, ProfilingRequestSet(),
                           &fill_value, sizeof(fill_value), wait_for);
      }

      TransposeExperiment<N> *exp = new TransposeExperiment<N>;
      exp->src_perm = &perms[i];
      exp->dst_perm = &perms[j];
      exp->nanoseconds = 0;
      experiments.push_back(exp);

      UserEvent done = UserEvent::create_user_event();
      done_events.push_back(done);

      CopyProfResult cpr;
      cpr.nanoseconds = &(exp->nanoseconds);
      cpr.done = done;

      ProfilingRequestSet prs;
      prs.add_request(prof_proc, COPYPROF_TASK, &cpr, sizeof(CopyProfResult))
	.add_measurement<ProfilingMeasurements::OperationTimeline>();
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].inst = src_inst;
      srcs[0].field_id = 0;
      srcs[0].size = field_sizes[0];
      dsts[0].inst = dst_inst;
      dsts[0].field_id = 0;
      dsts[0].size = field_sizes[0];
      wait_for = is.copy(srcs, dsts, prs, wait_for);

      dst_inst.destroy(wait_for);
    }

    src_inst.destroy(wait_for);
  }

  // wait for copies to finish
  done_events.push_back(wait_for);
  Event::merge_events(done_events).wait();

  for(typename std::vector<TransposeExperiment<N> *>::const_iterator it = experiments.begin();
      it != experiments.end();
      ++it) {
    double bw = 1.0 * is.volume() * field_sizes[0] / (*it)->nanoseconds;
    log_app.print() << "src=" << (*it)->src_perm->name
		    << " dst=" << (*it)->dst_perm->name
		    << " time=" << (1e-9 * (*it)->nanoseconds)
		    << " bw=" << bw;
    delete *it;
  }
}

std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Copy/transpose test";

  size_t log2_buffer_size = 0;
  {
    size_t v = TestConfig::buffer_size / sizeof(int);
    if(TestConfig::narrow_dim > 0)
      v /= TestConfig::narrow_dim;
    while(v > 1) {
      v >>= 1;
      log2_buffer_size++;
    }
  }

  std::vector<Memory> src_mems, dst_mems;
  Machine::MemoryQuery mq(Machine::get_machine());
  for(Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it) {
    // skip small memories
    if((*it).capacity() < (3 * TestConfig::buffer_size)) continue;

    // only sysmem if were not doing all combinations
    if(((*it).kind() != Memory::SYSTEM_MEM) && !TestConfig::all_mems)
      continue;

    src_mems.push_back(*it);
  }
  assert(!src_mems.empty());

  if(TestConfig::all_mems) {
    dst_mems = src_mems;
  } else {
    // just use first as source and last as dest (gives you inter-node
    //  copies when multiple nodes are present)
    dst_mems.push_back(src_mems[src_mems.size() - 1]);
    src_mems.resize(1);
  }

  for(std::vector<Memory>::const_iterator src_it = src_mems.begin();
      src_it != src_mems.end();
      ++src_it)
    for(std::vector<Memory>::const_iterator dst_it = dst_mems.begin();
        dst_it != dst_mems.end();
        ++dst_it) {
      log_app.print() << "srcmem=" << *src_it << " dstmem=" << *dst_it;
      typedef int FT;
      if((TestConfig::dim_mask & 1) != 0)
        do_single_dim<1, FT>(*src_it, *dst_it, log2_buffer_size, 0, p);
      if((TestConfig::dim_mask & 2) != 0)
        do_single_dim<2, FT>(*src_it, *dst_it, log2_buffer_size,
                             TestConfig::narrow_dim, p);
      if((TestConfig::dim_mask & 4) != 0)
        do_single_dim<3, FT>(*src_it, *dst_it, log2_buffer_size,
                             TestConfig::narrow_dim, p);
    }
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int_units("-b", TestConfig::buffer_size, 'M')
    .add_option_int("-narrow", TestConfig::narrow_dim)
    .add_option_int("-dims", TestConfig::dim_mask)
    .add_option_int("-all", TestConfig::all_mems);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
				   COPYPROF_TASK,
				   CodeDescriptor(copy_profiling_task),
				   ProfilingRequestSet(),
				   0, 0).wait();

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
