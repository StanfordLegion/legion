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
  WRITER_TASK,
  READER_TASK,
};

enum {
  FID_DATA = 100,
};

struct WriterTaskArgs {
  IndexSpace<1> is;
  RegionInstance inst;
  int wrval;
};

struct ReaderTaskArgs {
  IndexSpace<1> is;
  RegionInstance inst;
  int rdval;
};

void writer_task(const void *args, size_t arglen, 
		 const void *userdata, size_t userlen, Processor p)
{
  const WriterTaskArgs& wargs = *reinterpret_cast<const WriterTaskArgs *>(args);
  AffineAccessor<int, 1> acc(wargs.inst, FID_DATA);
  for(IndexSpaceIterator<1> it(wargs.is); it.valid; it.step())
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      acc[it2.p] = it2.p.x + wargs.wrval;
    }
}

int correct = 0;

void reader_task(const void *args, size_t arglen, 
		 const void *userdata, size_t userlen, Processor p)
{
  const ReaderTaskArgs& rargs = *reinterpret_cast<const ReaderTaskArgs *>(args);
  AffineAccessor<int, 1> acc(rargs.inst, FID_DATA);
  for(IndexSpaceIterator<1> it(rargs.is); it.valid; it.step())
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      int expval = it2.p.x + rargs.rdval;
      int actval = acc[it2.p];
      if(expval == actval)
	correct++;
      else
	log_app.error() << "MISMATCH: " << it2.p << ": " << actval << " != " << expval;
    }
}

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Realm subgraphs test";

  // do everything on this processor - get a good memory to use
  Memory m = Machine::MemoryQuery(Machine::get_machine()).has_affinity_to(p).first();
  assert(m.exists());

  IndexSpace<1> is = Rect<1>(0, 9);
  RegionInstance inst1, inst2;
  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_DATA] = sizeof(int);

  Event e = RegionInstance::create_instance(inst1, m, is, field_sizes,
					    0 /*block_size=SOA*/,
					    ProfilingRequestSet());

  e = RegionInstance::create_instance(inst2, m, is, field_sizes,
				      0 /*block_size=SOA*/,
				      ProfilingRequestSet(), e);

  // immediate mode - no subgraph
  WriterTaskArgs w_args;
  w_args.is = is;
  w_args.inst = inst1;
  w_args.wrval = 5;
  e = p.spawn(WRITER_TASK, &w_args, sizeof(w_args), e);

  std::vector<CopySrcDstField> copy_src(1), copy_dst(1);
  copy_src[0].set_field(inst1, FID_DATA, sizeof(int));
  copy_dst[0].set_field(inst2, FID_DATA, sizeof(int));
  e = is.copy(copy_src, copy_dst, ProfilingRequestSet(), e);

  ReaderTaskArgs r_args;
  r_args.is = is;
  r_args.inst = inst2;
  r_args.rdval = 5;
  e = p.spawn(READER_TASK, &r_args, sizeof(r_args), e);
 
  e.wait();

  Reservation rsrv = Reservation::create_reservation();

  SubgraphDefinition sd;
  sd.tasks.resize(2);
  sd.tasks[0].proc = p;
  sd.tasks[0].task_id = WRITER_TASK;
  w_args.wrval = 6;
  sd.tasks[0].args.set(&w_args, sizeof(w_args));
  sd.tasks[1].proc = p;
  sd.tasks[1].task_id = READER_TASK;
  r_args.rdval = 7;
  sd.tasks[1].args.set(&r_args, sizeof(r_args));

  sd.copies.resize(1);
  sd.copies[0].space = is;
  sd.copies[0].srcs = copy_src;
  sd.copies[0].dsts = copy_dst;

  sd.acquires.resize(1);
  sd.acquires[0].rsrv = rsrv;
  
  sd.releases.resize(1);
  sd.releases[0].rsrv = rsrv;

  sd.dependencies.resize(6);
  sd.dependencies[0].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[0].src_op_index = 0;
  sd.dependencies[0].tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
  sd.dependencies[0].tgt_op_index = 0;

  sd.dependencies[1].src_op_kind = SubgraphDefinition::OPKIND_COPY;
  sd.dependencies[1].src_op_index = 0;
  sd.dependencies[1].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[1].tgt_op_index = 1;

  sd.dependencies[2].src_op_kind = SubgraphDefinition::OPKIND_ACQUIRE;
  sd.dependencies[2].src_op_index = 0;
  sd.dependencies[2].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[2].tgt_op_index = 0;

  sd.dependencies[3].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[3].src_op_index = 1;
  sd.dependencies[3].tgt_op_kind = SubgraphDefinition::OPKIND_RELEASE;
  sd.dependencies[3].tgt_op_index = 0;

  sd.dependencies[4].src_op_kind = SubgraphDefinition::OPKIND_EXT_PRECOND;
  sd.dependencies[4].src_op_index = 0;
  sd.dependencies[4].tgt_op_kind = SubgraphDefinition::OPKIND_ACQUIRE;
  sd.dependencies[4].tgt_op_index = 0;

  sd.dependencies[5].src_op_kind = SubgraphDefinition::OPKIND_ACQUIRE;
  sd.dependencies[5].src_op_index = 0;
  sd.dependencies[5].tgt_op_kind = SubgraphDefinition::OPKIND_EXT_POSTCOND;
  sd.dependencies[5].tgt_op_index = 0;

  sd.interpolations.resize(2);
  sd.interpolations[0].bytes = sizeof(int);
  sd.interpolations[0].target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
  sd.interpolations[0].target_index = 0;
  // clang doesn't permit use of offsetof here because IndexSpace<N,T> is non-POD
  size_t wrval_offset = reinterpret_cast<size_t>(&w_args.wrval) - reinterpret_cast<size_t>(&w_args);
  sd.interpolations[0].target_offset = wrval_offset;

  sd.interpolations[1].bytes = sizeof(int);
  sd.interpolations[1].target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
  sd.interpolations[1].target_index = 1;
  // clang doesn't permit use of offsetof here because IndexSpace<N,T> is non-POD
  size_t rdval_offset = reinterpret_cast<size_t>(&r_args.rdval) - reinterpret_cast<size_t>(&r_args);
  sd.interpolations[1].target_offset = rdval_offset;

  Subgraph sg;
  e = Subgraph::create_subgraph(sg,
				sd,
				ProfilingRequestSet());

  std::vector<UserEvent> start_events(4);
  std::vector<Event> acquire_events(4);
  std::vector<Event> finish_events(4);
  for(int i = 0; i < 4; i++) {
    int val = 100 * (i + 1);
    start_events[i] = UserEvent::create_user_event();
    std::vector<Event> preconds(1, start_events[i]);
    std::vector<Event> postconds(1);
    finish_events[i] = sg.instantiate(&val, sizeof(val),
				      ProfilingRequestSet(),
				      preconds,
				      postconds,
				      e);
    acquire_events[i] = postconds[0];
  }
  e = Event::merge_events(finish_events);

  // connect up start events
  start_events[0].trigger(acquire_events[1]);
  start_events[1].trigger(acquire_events[2]);
  start_events[2].trigger(acquire_events[3]);
  start_events[3].trigger();

  sg.destroy(e);

  e.wait();

  int expcorrect = 50;
  log_app.info() << correct << " correct comparisons (out of " << expcorrect << ")";
  bool ok = (correct == expcorrect);
  Runtime::get_runtime().shutdown(Event::NO_EVENT,
				  ok ? 0 : 1);
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
  rt.register_task(WRITER_TASK, writer_task);
  rt.register_task(READER_TASK, reader_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single main task
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // main task will call shutdown - wait for that and return the exit code
  return rt.wait_for_shutdown();
}
