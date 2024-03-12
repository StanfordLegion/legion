/* Copyright 2024 Stanford University
 * Copyright 2024 Los Alamos National Laboratory
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

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <set>
#include <time.h>

#include <realm.h>

using namespace Realm;

typedef long long coord_t;

// TASK IDs
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  HIST_BATCH_TASK  = Processor::TASK_ID_FIRST_AVAILABLE+1,
  HIST_BATCH_LOCALIZE_TASK  = Processor::TASK_ID_FIRST_AVAILABLE+2,
  HIST_BATCH_REDFOLD_TASK  = Processor::TASK_ID_FIRST_AVAILABLE+3, 
  HIST_BATCH_REDLIST_TASK  = Processor::TASK_ID_FIRST_AVAILABLE+4,
  HIST_BATCH_REDSINGLE_TASK  = Processor::TASK_ID_FIRST_AVAILABLE+5,
};

// reduction op IDs
enum {
  REDOP_BUCKET_ADD = 1,
};

Logger log_app("appl");

template <bool EXCL, class LHS, class RHS>
struct DoAdd {
  static void do_add(LHS& lhs, RHS rhs);
};

template <class LHS, class RHS>
struct DoAdd<true,LHS,RHS> {
  static void do_add(LHS& lhs, RHS rhs)
  {
    lhs += rhs;
  }
};

template <class LHS, class RHS>
struct DoAdd<false,LHS,RHS> {
  static void do_add(LHS& lhs, RHS rhs)
  {
    __sync_fetch_and_add(&lhs, (LHS)rhs);
  }
};

template <class LTYPE, class RTYPE>
struct ReductionAdd {
  typedef LTYPE LHS;
  typedef RTYPE RHS;
  template <bool EXCL> 
  static void apply(LTYPE& lhs, RTYPE rhs)
  {
    DoAdd<EXCL,LTYPE,RTYPE>::do_add(lhs, rhs);
  }
  static const RTYPE identity;
  template <bool EXCL> 
  static void fold(RTYPE& rhs1, RTYPE rhs2)
  {
    DoAdd<EXCL,RTYPE,RTYPE>::do_add(rhs1, rhs2);
  }
};

template <class LTYPE, class RTYPE>
/*static*/ const RTYPE ReductionAdd<LTYPE,RTYPE>::identity = 0;

/*
template <class LTYPE, class RTYPE>
template <>
static void ReductionAdd::apply<false>(LTYPE& lhs, RTYPE rhs)
{
  lhs += rhs;
}
*/
struct InputArgs {
  int argc;
  char **argv;
};

typedef unsigned BucketType;
typedef ReductionAdd<BucketType, int> BucketReduction;

template <class T>
struct HistBatchArgs {
  unsigned start, count;
  IndexSpace<1, coord_t> region;
  RegionInstance inst;
  Reservation lock;
  unsigned buckets;
  unsigned seed1, seed2;
};
  
InputArgs& get_input_args(void)
{
  static InputArgs args;
  return args;
}

static Memory closest_memory(Processor p)
{
  std::vector<Machine::ProcessorMemoryAffinity> pmas;
  Machine::get_machine().get_proc_mem_affinity(pmas, p);

  assert(pmas.size() > 0);
  Memory m = pmas[0].m;
  unsigned best_lat = pmas[0].latency;

  for(size_t i = 1; i < pmas.size(); i++) {
    // ignore memories that have no capacity - can't create instances there
    if(pmas[i].m.capacity() == 0) continue;

    if(pmas[i].latency < best_lat) {
      m = pmas[i].m;
      best_lat = pmas[i].latency;
    }
  }

  return m;
}

static Memory farthest_memory(Processor p)
{
  std::vector<Machine::ProcessorMemoryAffinity> pmas;
  Machine::get_machine().get_proc_mem_affinity(pmas, p);

  assert(pmas.size() > 0);
  Memory m = pmas[0].m;
  unsigned worst_lat = pmas[0].latency;

  for(size_t i = 1; i < pmas.size(); i++) {
    // ignore memories that have no capacity - can't create instances there
    if(pmas[i].m.capacity() == 0) continue;

    if(pmas[i].latency > worst_lat) {
      m = pmas[i].m;
      worst_lat = pmas[i].latency;
    }
  }

  return m;
}

static void run_case(const char *name, int task_id,
		     HistBatchArgs<BucketType>& hbargs, int num_batches,
		     bool use_lock)
{
  // clear histogram
#if 0
  if(0) {
    RegionAccessor<AccessorType::Generic,BucketType> ria = hbargs.inst.get_accessor().typeify<BucketType>();

    for(unsigned i = 0; i < hbargs.buckets; i++)
      ria.write(ptr_t(i), 0);
  }
#endif

  log_app.info("starting %s histogramming...\n", name);

  double start_time = Realm::Clock::current_time_in_microseconds();

  // now do the histogram
  std::set<Event> batch_events;
  std::set<Processor> all_procs;
  Machine::get_machine().get_all_processors(all_procs);
  assert(all_procs.size() > 0);
  std::set<Processor>::const_iterator it = all_procs.begin();
  for(int i = 0; i < num_batches; i++) {
    hbargs.start = i * hbargs.count;

    if(it == all_procs.end()) it = all_procs.begin();
    Processor tgt = *(it++);
    log_app.debug("sending batch %d to processor " IDFMT "\n", i, tgt.id);

    Event wait_for;
    if(use_lock)
      wait_for = hbargs.lock.acquire();
    else
      wait_for = Event::NO_EVENT;

    Event e = tgt.spawn(task_id, &hbargs, sizeof(hbargs), wait_for);
    batch_events.insert(e);

    if(use_lock)
      hbargs.lock.release(e);
  }

  Event all_done = Event::merge_events(batch_events);

  log_app.info("waiting for batches to finish...\n");
  all_done.wait();

  double end_time = Realm::Clock::current_time_in_microseconds();
  log_app.info("done\n");
  printf("ELAPSED(%s) = %f\n", name, (end_time - start_time)*1e-6);
}		     

void top_level_task(const void *args, size_t arglen, 
                    const void *userdata, size_t userlen, Processor p)
{
  int buckets = 1048576;
  int num_batches = 100;
  int batch_size = 1048576;
  int seed1 = 12345;
  int seed2 = 54321;
  int do_slow = 0;

  // Parse the input arguments
#define INT_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = atoi((argv)[++i]);		\
          continue;					\
        } } while(0)

#define BOOL_ARG(argname, varname) do { \
        if(!strcmp((argv)[i], argname)) {		\
          varname = true;				\
          continue;					\
        } } while(0)
  {
    InputArgs &inputs = get_input_args();
    char **argv = inputs.argv;
    for (int i = 1; i < inputs.argc; i++)
    {
      INT_ARG("-doslow", do_slow);
      INT_ARG("-buckets", buckets);
      INT_ARG("-batches", num_batches);
      INT_ARG("-bsize", batch_size);
    }
  }
#undef INT_ARG
#undef BOOL_ARG

  //UserEvent start_event = UserEvent::create_user_event();

  IndexSpace<1, coord_t> hist_region = Rect<1, coord_t>(0, buckets - 1);

  Reservation lock = Reservation::create_reservation();

  Memory m = farthest_memory(p);
  printf("placing master instance in memory " IDFMT "\n", m.id);
  RegionInstance hist_inst;
  RegionInstance::create_instance(hist_inst, m, hist_region,
				  std::vector<size_t>(1, sizeof(BucketType)),
				  0, // SOA
				  ProfilingRequestSet()).wait();
  assert(hist_inst.exists());

  HistBatchArgs<BucketType> hbargs;
  hbargs.count = batch_size;
  hbargs.region = hist_region;
  hbargs.inst = hist_inst;
  hbargs.lock = lock;
  hbargs.buckets = buckets;
  hbargs.seed1 = seed1;
  hbargs.seed2 = seed2;

  if(do_slow)
    run_case("original", HIST_BATCH_TASK, hbargs, num_batches, true);
  run_case("redfold", HIST_BATCH_REDFOLD_TASK, hbargs, num_batches, false);
  run_case("localize", HIST_BATCH_LOCALIZE_TASK, hbargs, num_batches, true);
  run_case("redlist", HIST_BATCH_REDLIST_TASK, hbargs, num_batches, false);
  if(do_slow)
    run_case("redsingle", HIST_BATCH_REDSINGLE_TASK, hbargs, num_batches, false);

#if 0
  {
    RegionInstanceAccessor<BucketType,AccessorGeneric> ria = hist_inst.get_accessor();

    for(int i = 0; i < buckets; i++)
      ria.write(ptr_t<BucketType>(i), 0);
  }

  HistBatchArgs<BucketType> hbargs;
  hbargs.count = batch_size;
  hbargs.region = hist_region;
  hbargs.inst = hist_inst;
  hbargs.buckets = buckets;
  hbargs.seed1 = seed1;
  hbargs.seed2 = seed2;

  run_case("original", HIST_BATCH_TASK, hbargs, num_batches);

  printf("starting histogramming...\n");

  double start_time = Clock::abs_time();

  // now do the histogram
  std::set<Event> batch_events;
  const std::set<Processor>& all_procs = Machine::get_machine()->get_all_processors();
  assert(all_procs.size() > 0);
  std::set<Processor>::const_iterator it = all_procs.begin();
  for(int i = 0; i < num_batches; i++) {
    HistBatchArgs<BucketType> hbargs;
    hbargs.start = i * batch_size;
    hbargs.count = batch_size;
    hbargs.region = hist_region;
    hbargs.inst = hist_inst;
    hbargs.buckets = buckets;
    hbargs.seed1 = seed1;
    hbargs.seed2 = seed2;

    if(it == all_procs.end()) it = all_procs.begin();
    Processor tgt = *(it++);
    printf("sending batch %d to processor %x\n", i, tgt.id);

    Event e = tgt.spawn(HIST_BATCH_TASK, &hbargs, sizeof(hbargs));
    batch_events.insert(e);
  }

  Event all_done = Event::merge_events(batch_events);

  printf("waiting for batches to finish...\n");
  all_done.wait();

  double end_time = Clock::abs_time();
  printf("done\n");
  printf("ELAPSED = %f\n", end_time - start_time);
#endif
}

static unsigned myrand(unsigned long long ival,
		       unsigned seed1, unsigned seed2)
{
  unsigned long long rstate = ival;
  for(int j = 0; j < 16; j++) {
    rstate = (0x5DEECE66DULL * rstate + 0xB) & 0xFFFFFFFFFFFFULL;
    rstate ^= (((ival >> j) & 1) ? seed1 : seed2);
  }
  return rstate;
}

template <class REDOP>
void hist_batch_task(const void *args, size_t arglen, 
                     const void *userdata, size_t userlen, Processor p)
{
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // get a reduction accessor for the instance
  AffineAccessor<BucketType, 1, coord_t> ria(hbargs->inst, 0 /*field id*/);

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    coord_t bucket = rval % hbargs->buckets;

    REDOP::template apply<false>(ria[bucket], 1);
  }
}
  
template <class REDOP>
void hist_batch_localize_task(const void *args, size_t arglen, 
                              const void *userdata, size_t userlen, Processor p)
{
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // create a local full instance
  Memory m = closest_memory(p);
  RegionInstance lclinst = RegionInstance::NO_INST;
  RegionInstance::create_instance(lclinst, m, hbargs->region,
				  std::vector<size_t>(1, sizeof(BucketType)),
				  0, // SOA
				  ProfilingRequestSet()).wait();
  assert(lclinst.exists());

  std::vector<CopySrcDstField> src(1);
  src[0].inst = hbargs->inst;
  src[0].field_id = 0;
  src[0].size = sizeof(BucketType);
  std::vector<CopySrcDstField> dst(1);
  dst[0].inst = lclinst;
  dst[0].field_id = 0;
  dst[0].size = sizeof(BucketType);
  hbargs->region.copy(src, dst, 
		      ProfilingRequestSet()).wait();

  // get an array accessor for the instance
  AffineAccessor<BucketType, 1, coord_t> ria(lclinst, 0 /*field id*/);

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    coord_t bucket = rval % hbargs->buckets;

    REDOP::template apply<false>(ria[bucket], 1);
  }

  // now copy the local instance back to the original one
  Event done = hbargs->region.copy(dst, src,
				   ProfilingRequestSet());

  lclinst.destroy(done);

  done.wait();
}
  
template <class REDOP>
void hist_batch_redfold_task(const void *args, size_t arglen, 
                             const void *userdata, size_t userlen, Processor p)
{
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // create a reduction fold instance
  Memory m = closest_memory(p);
  RegionInstance redinst = RegionInstance::NO_INST;
  RegionInstance::create_instance(redinst, m, hbargs->region,
				  typename std::vector<size_t>(1, sizeof(BucketReduction::RHS)),
				  0, // SOA
				  ProfilingRequestSet()).wait();
  assert(redinst.exists());

  // clear the instance
  std::vector<CopySrcDstField> fld(1);
  fld[0].inst = redinst;
  fld[0].field_id = 0;
  fld[0].size = sizeof(BucketReduction::RHS);
  hbargs->region.fill(fld,
		      ProfilingRequestSet(),
		      &BucketReduction::identity, fld[0].size).wait();

  // get a reduction accessor for the instance
  AffineAccessor<BucketReduction::RHS, 1, coord_t> ria(redinst, 0 /*field id*/);

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    coord_t bucket = rval % hbargs->buckets;

    REDOP::template fold<false>(ria[bucket], 1);
  }

  // now copy the reduction instance back to the original one
  std::vector<CopySrcDstField> src(1);
  src[0].inst = redinst;
  src[0].field_id = 0;
  src[0].size = sizeof(BucketReduction::RHS);
  std::vector<CopySrcDstField> dst(1);
  dst[0].inst = hbargs->inst;
  dst[0].field_id = 0;
  dst[0].size = sizeof(BucketType);
  dst[0].set_redop(REDOP_BUCKET_ADD, false /*!fold*/);
  Event done = hbargs->region.copy(src, dst, 
				   ProfilingRequestSet(),
				   Event::NO_EVENT);

  redinst.destroy(done);

  done.wait();
}

template <class REDOP>
void hist_batch_redlist_task(const void *args, size_t arglen, 
                             const void *userdata, size_t userlen, Processor p)
{
#if 0
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // create a reduction list instance
  Memory m = closest_memory(p);
  RegionInstance<BucketType> redinst = RegionInstance::NO_INST;
  while (!redinst.exists())
    redinst = hbargs->region.create_instance(m, REDOP_BUCKET_ADD, hbargs->count, hbargs->inst);
  assert(redinst.exists());

  // get a reduction accessor for the instance
  RegionInstanceAccessor<BucketType,AccessorReductionList> ria = redinst.get_accessor().convert<AccessorReductionList>();

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    unsigned bucket = rval % hbargs->buckets;

    ria.reduce<REDOP>(ptr_t<BucketType>(bucket), 1);
  }

  // now copy the reduction instance back to the original one
  redinst.copy_to(hbargs->inst).wait();

  hbargs->region.destroy_instance(redinst);
#endif
}
  
template <class REDOP>
void hist_batch_redsingle_task(const void *args, size_t arglen, 
                               const void *userdata, size_t userlen, Processor p)
{
#if 0
  const HistBatchArgs<BucketType> *hbargs = (const HistBatchArgs<BucketType> *)args;

  // create a reduction list instance
  Memory m = closest_memory(p);
  RegionInstance<BucketType> redinst = RegionInstance::NO_INST;
  while (!redinst.exists())
    redinst = hbargs->region.create_instance(m, REDOP_BUCKET_ADD, 1, hbargs->inst);
  assert(redinst.exists());

  // get a reduction accessor for the instance
  RegionInstanceAccessor<BucketType,AccessorReductionList> ria = redinst.get_accessor().convert<AccessorReductionList>();

  for(unsigned i = 0; i < hbargs->count; i++) {
    unsigned rval = myrand(hbargs->start + i, hbargs->seed1, hbargs->seed2);
    unsigned bucket = rval % hbargs->buckets;

    ria.reduce<REDOP>(ptr_t<BucketType>(bucket), 1);

    // now copy the reduction instance back to the original one after each entry
    redinst.copy_to(hbargs->inst).wait();
  }

  hbargs->region.destroy_instance(redinst);
#endif
}
  
int main(int argc, char **argv)
{
  Runtime r;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  r.register_task(TOP_LEVEL_TASK, top_level_task);
  r.register_task(HIST_BATCH_TASK, hist_batch_task<BucketReduction>);
  r.register_task(HIST_BATCH_LOCALIZE_TASK, hist_batch_localize_task<BucketReduction>);
  r.register_task(HIST_BATCH_REDFOLD_TASK, hist_batch_redfold_task<BucketReduction>);
  r.register_task(HIST_BATCH_REDLIST_TASK, hist_batch_redlist_task<BucketReduction>);
  r.register_task(HIST_BATCH_REDSINGLE_TASK, hist_batch_redsingle_task<BucketReduction>);
  r.register_reduction(REDOP_BUCKET_ADD, ReductionOpUntyped::create_reduction_op<BucketReduction>());

  // Set the input args
  get_input_args().argv = argv;
  get_input_args().argc = argc;

  // select a processor to run the top level task on
  Processor p = Processor::NO_PROC;
  {
    std::set<Processor> all_procs;
    Machine::get_machine().get_all_processors(all_procs);
    for(std::set<Processor>::const_iterator it = all_procs.begin();
	it != all_procs.end();
	it++)
      if(it->kind() == Processor::LOC_PROC) {
	p = *it;
	break;
      }
  }
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = r.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  r.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  r.wait_for_shutdown();
  
  return 0;
}

