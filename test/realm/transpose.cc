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
			 int pos)
{
  static const char *dim_names = "XYZW";

  for(int i = 0; i < N; i++) {
    bool found = false;
    for(int j = 0; j < pos; j++)
      if(scratch.dim_order[j] == i) found = true;
    if(found) continue;
    
    scratch.dim_order[pos] = i;
    scratch.name[pos] = dim_names[i];

    if(pos == (N - 1))
      perms.push_back(scratch);
    else
      create_permutations<N>(perms, scratch, pos+1);
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

static size_t log2_buffer_size = 20; // should be bigger than any cache in system

template <int N, typename FT>
void do_single_dim(Memory src_mem, Memory dst_mem, int log2_size,
		   Processor prof_proc, int pad = 0)
{
  std::vector<LayoutPermutation<N> > perms;
  LayoutPermutation<N> scratch;
  memset(&scratch, 0, sizeof(scratch));
  create_permutations<N>(perms, scratch, 0);

  Rect<N> bounds;
  for(int i = 0; i < N; i++) {
    bounds.lo[i] = 0;
    bounds.hi[i] = (1 << (log2_size / N)) - 1;
  }
  IndexSpace<N> is(bounds);

  Rect<N> bounds_pad;
  for(int i = 0; i < N; i++) {
    bounds_pad.lo[i] = 0;
    bounds_pad.hi[i] = (1 << (log2_size / N)) - 1 + 32;
  }
  IndexSpace<N> is_pad(bounds_pad);

  std::vector<RegionInstance> src_insts, dst_insts;

  std::map<FieldID, size_t> field_sizes;
  field_sizes[0] = sizeof(FT);
  InstanceLayoutConstraints ilc(field_sizes, 1);

  for(typename std::vector<LayoutPermutation<N> >::const_iterator it = perms.begin();
      it != perms.end();
      ++it) {
    // src mem
    {
      InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout(is_pad, ilc, it->dim_order);
      RegionInstance s_inst;
      Event e = RegionInstance::create_instance(s_inst, src_mem, ilg,
						ProfilingRequestSet());
      std::vector<CopySrcDstField> tgt(1);
      tgt[0].inst = s_inst;
      tgt[0].field_id = 0;
      tgt[0].size = field_sizes[0];
      FT fill_value = 77;
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();
      src_insts.push_back(s_inst);
    }

    // dst mem
    {
      InstanceLayoutGeneric *ilg = InstanceLayoutGeneric::choose_instance_layout(is, ilc, it->dim_order);
      RegionInstance d_inst;
      Event e = RegionInstance::create_instance(d_inst, dst_mem, ilg,
						ProfilingRequestSet());
      std::vector<CopySrcDstField> tgt(1);
      tgt[0].inst = d_inst;
      tgt[0].field_id = 0;
      tgt[0].size = field_sizes[0];
      FT fill_value = 88;
      e = is.fill(tgt, ProfilingRequestSet(), &fill_value, sizeof(fill_value), e);
      e.wait();
      dst_insts.push_back(d_inst);
    }
  }

  std::vector<TransposeExperiment<N> *> experiments;
  std::set<Event> done_events;

  Event prev_copy = Event::NO_EVENT;
  for(unsigned i = 0; i < perms.size(); i++)
    for(unsigned j = 0; j < perms.size(); j++) {
      TransposeExperiment<N> *exp = new TransposeExperiment<N>;
      exp->src_perm = &perms[i];
      exp->dst_perm = &perms[j];
      exp->nanoseconds = 0;
      experiments.push_back(exp);

      UserEvent done = UserEvent::create_user_event();
      done_events.insert(done);

      CopyProfResult cpr;
      cpr.nanoseconds = &(exp->nanoseconds);
      cpr.done = done;

      ProfilingRequestSet prs;
      prs.add_request(prof_proc, COPYPROF_TASK, &cpr, sizeof(CopyProfResult))
	.add_measurement<ProfilingMeasurements::OperationTimeline>();
      std::vector<CopySrcDstField> srcs(1), dsts(1);
      srcs[0].inst = src_insts[i];
      srcs[0].field_id = 0;
      srcs[0].size = field_sizes[0];
      dsts[0].inst = dst_insts[j];
      dsts[0].field_id = 0;
      dsts[0].size = field_sizes[0];
      prev_copy = is.copy(srcs, dsts, prs, prev_copy);
    }

  // wait for copies to finish
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

  // cleanup
  for(typename std::vector<RegionInstance>::const_iterator it = src_insts.begin();
      it != src_insts.end();
      ++it)
    it->destroy();

  for(typename std::vector<RegionInstance>::const_iterator it = dst_insts.begin();
      it != dst_insts.end();
      ++it)
    it->destroy();
}

std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Copy/transpose test";

  // just sysmem for now
  Machine machine = Machine::get_machine();
  Memory m = Machine::MemoryQuery(machine).only_kind(Memory::SYSTEM_MEM).first();
  assert(m.exists());

  typedef int FT;
  do_single_dim<1, FT>(m, m, log2_buffer_size, p);
  do_single_dim<2, FT>(m, m, log2_buffer_size, p);
  do_single_dim<3, FT>(m, m, log2_buffer_size, p);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-b")) {
      log2_buffer_size = strtoll(argv[++i], 0, 10);
      continue;
    }

  }

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
