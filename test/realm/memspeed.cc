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
  FID_BASE = 44,
};

struct SpeedTestArgs {
  Memory mem;
  RegionInstance inst;
  size_t elements;
  int reps;
  Machine::AffinityDetails affinity;
};

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
  bool do_tasks = true;   // should tasks accessing memories be tested
  bool do_copies = true;  // should DMAs between memories be tested
  int copy_reps = 0;      // if nonzero, average over #reps copies
  int copy_fields = 1;    // number of distinct fields to copy
  size_t sparse_chunk = 0;  // if nonzero, test sparse copies with chunk size
  size_t sparse_gap = 16;   // gap between sparse chunks (if used)
  bool copy_aos = false;   // if true, use an AOS memory layout
  bool slow_mems = false;  // show slow memories be tested?
};

void memspeed_cpu_task(const void *args, size_t arglen, 
		       const void *userdata, size_t userlen, Processor p)
{
  const SpeedTestArgs& cargs = *(const SpeedTestArgs *)args;

  AffineAccessor<void *, 1> ra(cargs.inst, 0);

  // sequential write test
  double seqwr_bw = 0;
  {
    long long t1 = Clock::current_time_in_nanoseconds();
    for(int j = 0; j < cargs.reps; j++)
      for(size_t i = 0; i < cargs.elements; i++)
	ra[i] = 0;
    long long t2 = Clock::current_time_in_nanoseconds();
    seqwr_bw = 1.0 * cargs.reps * cargs.elements * sizeof(void *) / (t2 - t1);
  }

  // sequential read test
  double seqrd_bw = 0;
  {
    long long t1 = Clock::current_time_in_nanoseconds();
    int errors = 0;
    for(int j = 0; j < cargs.reps; j++)
      for(size_t i = 0; i < cargs.elements; i++) {
	void *ptr = *(void * volatile *)&ra[i];
	if(ptr != 0) errors++;
      }
    long long t2 = Clock::current_time_in_nanoseconds();
    assert(errors == 0);
    seqrd_bw = 1.0 * cargs.reps * cargs.elements * sizeof(void *) / (t2 - t1);
  }

  // random write test
  double rndwr_bw = 0;
  std::vector<void *> last_ptrs;  // for checking latency test
  {
    // run on many fewer elements...
    size_t count = cargs.elements >> 8;
    // quadratic stepping via "acceleration" and "velocity"
    size_t a = 548191;
    size_t v = 24819;
    size_t p = 0;
    long long t1 = Clock::current_time_in_nanoseconds();
    for(int j = 0; j < cargs.reps; j++) {
      for(size_t i = 0; i < count; i++) {
	size_t prev = p;
	p = (p + v) % cargs.elements;
	v = (v + a) % cargs.elements;
	// wrapping would be bad
	assert(p != 0);
	ra[prev] = &ra[p];
      }
      last_ptrs.push_back(&ra[p]);
    }
    long long t2 = Clock::current_time_in_nanoseconds();
    rndwr_bw = 1.0 * cargs.reps * count * sizeof(void *) / (t2 - t1);
  } 

  // random read test
  double rndrd_bw = 0;
  {
    // run on many fewer elements...
    size_t count = cargs.elements >> 8;
    // quadratic stepping via "acceleration" and "velocity"
    size_t a = 548191;
    size_t v = 24819;
    size_t p = 0;
    long long t1 = Clock::current_time_in_nanoseconds();
    int errors = 0;
    for(int j = 0; j < cargs.reps; j++)
      for(size_t i = 0; i < count; i++) {
	size_t prev = p;
	p = (p + v) % cargs.elements;
	v = (v + a) % cargs.elements;
	// wrapping would be bad
	assert(p != 0);
	void *exp = &ra[p];
	void *act = ra[prev];
	if(exp != act) {
	  //log_app.info() << "error: " << exp << " != " << act;
	  errors++;
	}
      }
    long long t2 = Clock::current_time_in_nanoseconds();
    if(errors > 0)
      log_app.warning() << errors << " errors during random read test";
    rndrd_bw = 1.0 * cargs.reps * count * sizeof(void *) / (t2 - t1);
  } 

  // latency test
  double latency = 0;
  {
    // run on many fewer elements...
    size_t count = cargs.elements >> 8;
    // quadratic stepping via "acceleration" and "velocity"
    long long t1 = Clock::current_time_in_nanoseconds();
    int errors = 0;
    void **ptr = &ra[0];
    for(int j = 0; j < cargs.reps; j++) {
      for(size_t i = 0; i < count; i++)
	ptr = (void **)*ptr;
      assert(ptr == last_ptrs[j]);
    }
    long long t2 = Clock::current_time_in_nanoseconds();
    assert(errors == 0);
    latency = 1.0 * (t2 - t1) / (cargs.reps * count);
  } 

  log_app.info() << " on proc " << p << " seqwr:" << seqwr_bw << " seqrd:" << seqrd_bw;
  log_app.info() << " on proc " << p << " rndwr:" << rndwr_bw << " rndrd:" << rndrd_bw;
  log_app.info() << " on proc " << p << " latency:" << latency;
}

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
extern "C" {
  double gpu_seqwr_test(void *buffer, size_t reps, size_t elements);
  double gpu_seqrd_test(void *buffer, size_t reps, size_t elements);
  double gpu_rndwr_test(void *buffer, size_t reps, size_t elements);
  double gpu_rndrd_test(void *buffer, size_t reps, size_t elements);
  double gpu_latency_test(void *buffer, size_t reps, size_t elements);
}

void memspeed_gpu_task(const void *args, size_t arglen, 
		       const void *userdata, size_t userlen, Processor p)
{
  const SpeedTestArgs& cargs = *(const SpeedTestArgs *)args;

  AffineAccessor<void *, 1> ra(cargs.inst, 0);
  assert(ra.strides[0] == sizeof(void *));

  // sequential write test
  double seqwr_bw = gpu_seqwr_test(&ra[0], cargs.reps, cargs.elements);

  // sequential read test
  double seqrd_bw = gpu_seqrd_test(&ra[0], cargs.reps, cargs.elements);

  // random write test
  double rndwr_bw = gpu_rndwr_test(&ra[0], cargs.reps, cargs.elements);

  // random read test
  double rndrd_bw = gpu_rndrd_test(&ra[0], cargs.reps, cargs.elements);

  // latency test
  double latency = gpu_latency_test(&ra[0], cargs.reps, cargs.elements);

  log_app.info() << " on proc " << p << " seqwr:" << seqwr_bw << " seqrd:" << seqrd_bw;
  log_app.info() << " on proc " << p << " rndwr:" << rndwr_bw << " rndrd:" << rndrd_bw;
  log_app.info() << " on proc " << p << " latency:" << latency;
}
#endif

std::set<Processor::Kind> supported_proc_kinds;

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "Realm memory speed test";

  size_t elements = TestConfig::buffer_size / sizeof(void *);
  IndexSpace<1> d = Rect<1>(0, elements - 1);

  // build the list of memories that we want to test
  std::vector<Memory> memories;
  Machine machine = Machine::get_machine();
  for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it; ++it) {
    Memory m = *it;
    size_t capacity = m.capacity();
    // we need two instances if we're doing copy testing
    if(capacity < (TestConfig::buffer_size *
		   (TestConfig::do_copies ? 2*TestConfig::copy_fields : 1))) {
      log_app.info() << "skipping memory " << m << " (kind=" << m.kind() << ") - insufficient capacity";
      continue;
    }
    if((m.kind() == Memory::GLOBAL_MEM) && !TestConfig::slow_mems) {
      log_app.info() << "skipping memory " << m << " (kind=" << m.kind() << ") - slow global memory";
      continue;
    }
    if(ID(m).is_ib_memory()) {
      log_app.info() << "skipping memory " << m << " (kind=" << m.kind() << ") - intermediate buffer memory";
      continue;
    }

    log_app.print() << "Memory: " << m << " Kind:" << m.kind() << " Capacity: " << m.capacity();
    memories.push_back(m);
  }
  
  // iterate over memories, create an instance, and then let each processor beat on it
  if(TestConfig::do_tasks) {
    for(std::vector<Memory>::const_iterator it = memories.begin();
	it != memories.end();
	++it) {
      Memory m = *it;
      std::vector<size_t> field_sizes(1, sizeof(void *));
      RegionInstance inst;
      RegionInstance::create_instance(inst, m, d,
				      std::vector<size_t>(1, sizeof(void *)),
				      0 /*SOA*/,
				      ProfilingRequestSet()).wait();
      assert(inst.exists());

      // clear the instance first - this should also take care of faulting it in
      void *fill_value = 0;
      std::vector<CopySrcDstField> sdf(1);
      sdf[0].inst = inst;
      sdf[0].field_id = 0;
      sdf[0].size = sizeof(void *);
      d.fill(sdf, ProfilingRequestSet(), &fill_value, sizeof(fill_value)).wait();

      Machine::ProcessorQuery pq = Machine::ProcessorQuery(machine).has_affinity_to(m);
      for (Machine::ProcessorQuery::iterator it2 = pq.begin(); it2; ++it2) {
        Processor p = *it2;

        SpeedTestArgs cargs;
        cargs.mem = m;
        cargs.inst = inst;
        cargs.elements = elements;
        cargs.reps = 8;
        bool ok = machine.has_affinity(p, m, &cargs.affinity);
        assert(ok);

        log_app.info() << "  Affinity: " << p << "->" << m
                       << " BW: " << cargs.affinity.bandwidth
                       << " Latency: " << cargs.affinity.latency;

        if (supported_proc_kinds.count(p.kind()) == 0) {
          log_app.info() << "processor " << p << " is of unsupported kind "
                         << p.kind() << " - skipping";
          continue;
        }

        Event e = cargs.inst.fetch_metadata(p);
        e = p.spawn(MEMSPEED_TASK, &cargs, sizeof(cargs), e);

        e.wait();
      }

      inst.destroy();
    }
  }

  if(TestConfig::do_copies) {
    std::map<FieldID, size_t> field_sizes;
    for(int i = 0; i < TestConfig::copy_fields; i++)
      field_sizes[FID_BASE + i] = sizeof(void *);

    // do we need a sparse index space?
    IndexSpace<1> d_sparse;
    size_t sparse_elements = 0;
    if(TestConfig::sparse_chunk > 0) {
      std::vector<Rect<1> > rects;
      for(size_t ofs = 0;
          ofs <= (elements - TestConfig::sparse_chunk);
          ofs += (TestConfig::sparse_chunk + TestConfig::sparse_gap)) {
        rects.push_back(Rect<1>(ofs, ofs + TestConfig::sparse_chunk - 1));
        sparse_elements += TestConfig::sparse_chunk;
      }
      d_sparse = IndexSpace<1>(rects);
    }
    
    for(std::vector<Memory>::const_iterator it = memories.begin();
	it != memories.end();
	++it) {
      Memory m1 = *it;

      RegionInstance inst1;
      RegionInstance::create_instance(inst1, m1, d, field_sizes,
				      (TestConfig::copy_aos ? 1 : 0),
				      ProfilingRequestSet()).wait();
      assert(inst1.exists());

      // clear the instance first - this should also take care of faulting it in
      {
        void *fill_value = 0;
        std::vector<CopySrcDstField> srcs(TestConfig::copy_fields);
        for(int i = 0; i < TestConfig::copy_fields; i++)
          srcs[i].set_fill(fill_value);
        std::vector<CopySrcDstField> dsts(TestConfig::copy_fields);
        for(int i = 0; i < TestConfig::copy_fields; i++)
          dsts[i].set_field(inst1, FID_BASE+i, sizeof(void *));

        d.copy(srcs, dsts, ProfilingRequestSet()).wait();
      }

      for(std::vector<Memory>::const_iterator it2 = memories.begin();
	it2 != memories.end();
	++it2) {
	Memory m2 = *it2;

	RegionInstance inst2;
	RegionInstance::create_instance(inst2, m2, d, field_sizes,
					(TestConfig::copy_aos ? 1 : 0),
					ProfilingRequestSet()).wait();
	assert(inst2.exists());

	// clear the instance first - this should also take care of faulting it in
        {
          void *fill_value = 0;
          std::vector<CopySrcDstField> srcs(TestConfig::copy_fields);
          for(int i = 0; i < TestConfig::copy_fields; i++)
            srcs[i].set_fill(fill_value);
          std::vector<CopySrcDstField> dsts(TestConfig::copy_fields);
          for(int i = 0; i < TestConfig::copy_fields; i++)
            dsts[i].set_field(inst2, FID_BASE+i, sizeof(void *));
          d.copy(srcs, dsts, ProfilingRequestSet()).wait();
        }

	long long total_full_copy_time = 0;
	long long total_short_copy_time = 0;
        long long total_sparse_copy_time = 0;

        std::vector<CopySrcDstField> srcs(TestConfig::copy_fields);
        for(int i = 0; i < TestConfig::copy_fields; i++)
          srcs[i].set_field(inst1, FID_BASE+i, sizeof(void *));
        std::vector<CopySrcDstField> dsts(TestConfig::copy_fields);
        for(int i = 0; i < TestConfig::copy_fields; i++)
          dsts[i].set_field(inst2, FID_BASE+i, sizeof(void *));

	for(int rep = 0; rep <= TestConfig::copy_reps; rep++) {
	  // now perform two instance-to-instance copies

	  // copy #1 - full copy
	  long long full_copy_time = -1;
	  UserEvent full_copy_done = UserEvent::create_user_event();
	  {
	    CopyProfResult result;
	    result.nanoseconds = &full_copy_time;
	    result.done = full_copy_done;
	    ProfilingRequestSet prs;
	    prs.add_request(p, COPYPROF_TASK, &result, sizeof(CopyProfResult))
	      .add_measurement<ProfilingMeasurements::OperationTimeline>();
	    d.copy(srcs, dsts, prs).wait();
	  }

	  // copy #2 - single-element copy
	  long long short_copy_time = -1;
	  UserEvent short_copy_done = UserEvent::create_user_event();
	  {
	    CopyProfResult result;
	    result.nanoseconds = &short_copy_time;
	    result.done = short_copy_done;
	    ProfilingRequestSet prs;
	    prs.add_request(p, COPYPROF_TASK, &result, sizeof(CopyProfResult))
	      .add_measurement<ProfilingMeasurements::OperationTimeline>();
	    Rect<1>(0, 0).copy(srcs, dsts, prs).wait();
	  }

	  // wait for both results
	  full_copy_done.wait();
	  short_copy_done.wait();

	  if((rep > 0) || (TestConfig::copy_reps == 0)) {
	    total_full_copy_time += full_copy_time;
	    total_short_copy_time += short_copy_time;
	  }

          // optional copy #3 - sparse copy
          if(TestConfig::sparse_chunk > 0) {
            long long sparse_copy_time = -1;
            UserEvent sparse_copy_done = UserEvent::create_user_event();
            {
              CopyProfResult result;
              result.nanoseconds = &sparse_copy_time;
              result.done = sparse_copy_done;
              ProfilingRequestSet prs;
              prs.add_request(p, COPYPROF_TASK, &result, sizeof(CopyProfResult))
                .add_measurement<ProfilingMeasurements::OperationTimeline>();
              d_sparse.copy(srcs, dsts, prs).wait();
            }

            sparse_copy_done.wait();

            if((rep > 0) || (TestConfig::copy_reps == 0))
              total_sparse_copy_time += sparse_copy_time;
          }
	}

	if(TestConfig::copy_reps > 1) {
	  total_full_copy_time /= TestConfig::copy_reps;
	  total_short_copy_time /= TestConfig::copy_reps;
          total_sparse_copy_time /= TestConfig::copy_reps;
	}

	// latency is estimated as time to perfom single copy
	double latency = total_short_copy_time;

	// bandwidth is estimated based on extra time taken by full copy
	double bw = (1.0 * elements * TestConfig::copy_fields * sizeof(void *) /
		     (total_full_copy_time - total_short_copy_time));

        if(TestConfig::sparse_chunk == 0) {
          log_app.info() << "copy " << m1 << " -> " << m2 << ": bw:" << bw << " lat:" << latency;
        } else {
          double sparse_bw = (1.0 * sparse_elements * TestConfig::copy_fields * sizeof(void *) /
                              (total_sparse_copy_time - total_short_copy_time));

          log_app.info() << "copy " << m1 << " -> " << m2 << ": bw:" << bw << " lat:" << latency << " sparse_bw:" << sparse_bw;
        }

	inst2.destroy();
      }

      inst1.destroy();
    }
  }

  // HACK: there's a shutdown race condition related to instance destruction
  usleep(100000);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int_units("-b", TestConfig::buffer_size, 'M')
    .add_option_int("-tasks", TestConfig::do_tasks)
    .add_option_int("-copies", TestConfig::do_copies)
    .add_option_int("-reps", TestConfig::copy_reps)
    .add_option_int("-fields", TestConfig::copy_fields)
    .add_option_int("-sparse", TestConfig::sparse_chunk)
    .add_option_int("-gap", TestConfig::sparse_gap)
    .add_option_int("-aos", TestConfig::copy_aos)
    .add_option_int("-slowmem", TestConfig::slow_mems);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
				   MEMSPEED_TASK,
				   CodeDescriptor(memspeed_cpu_task),
				   ProfilingRequestSet(),
				   0, 0).wait();
  supported_proc_kinds.insert(Processor::LOC_PROC);

  Processor::register_task_by_kind(Processor::UTIL_PROC, false /*!global*/,
				   MEMSPEED_TASK,
				   CodeDescriptor(memspeed_cpu_task),
				   ProfilingRequestSet(),
				   0, 0).wait();
  supported_proc_kinds.insert(Processor::UTIL_PROC);

  Processor::register_task_by_kind(Processor::IO_PROC, false /*!global*/,
				   MEMSPEED_TASK,
				   CodeDescriptor(memspeed_cpu_task),
				   ProfilingRequestSet(),
				   0, 0).wait();
  supported_proc_kinds.insert(Processor::IO_PROC);

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
				   MEMSPEED_TASK,
				   CodeDescriptor(memspeed_gpu_task),
				   ProfilingRequestSet(),
				   0, 0).wait();
  supported_proc_kinds.insert(Processor::TOC_PROC);
#endif

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
