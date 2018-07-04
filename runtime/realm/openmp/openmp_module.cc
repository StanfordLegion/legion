/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "realm/openmp/openmp_module.h"

#include "realm/openmp/openmp_threadpool.h"

#include "realm/numa/numasysif.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/proc_impl.h"
#include "realm/threads.h"
#include "realm/runtime_impl.h"
#include "realm/utils.h"

namespace Realm {

  Logger log_omp("openmp");

  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalOpenMPProcessor

  // this is nearly identical to a LocalCPUProcessor, but it asks for its thread(s)
  //  to run on the specified numa domain

  class LocalOpenMPProcessor : public LocalTaskProcessor {
  public:
    LocalOpenMPProcessor(Processor _me, int _numa_node,
			 int _num_threads, bool _fake_cpukind,
			 CoreReservationSet& crs, size_t _stack_size,
			 bool _force_kthreads);
    virtual ~LocalOpenMPProcessor(void);

    virtual void shutdown(void);

    virtual void execute_task(Processor::TaskFuncID func_id,
			      const ByteArrayRef& task_args);

  protected:
    int numa_node;
    int num_threads;
    ThreadPool *pool;
    std::vector<CoreReservation *> core_rsrvs;
  };

  LocalOpenMPProcessor::LocalOpenMPProcessor(Processor _me, int _numa_node,
					     int _num_threads,
					     bool _fake_cpukind,
					     CoreReservationSet& crs,
					     size_t _stack_size,
					     bool _force_kthreads)
    : LocalTaskProcessor(_me, (_fake_cpukind ? Processor::LOC_PROC :
			                       Processor::OMP_PROC))
    , numa_node(_numa_node)
    , num_threads(_num_threads)
  {
    pool = new ThreadPool(num_threads - 1);

    // master runs in a user threads if possible
    {
      CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_numa_domain(numa_node);
      params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
      params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(_stack_size);

      std::string name = stringbuilder() << "OMP" << numa_node << " proc " << _me << " (master)";

      CoreReservation *rsrv = new CoreReservation(name, crs, params);
      core_rsrvs.push_back(rsrv);

#ifdef REALM_USE_USER_THREADS
      if(!_force_kthreads) {
	UserThreadTaskScheduler *sched = new UserThreadTaskScheduler(me, *rsrv);
	// no config settings we want to tweak yet
	set_scheduler(sched);
      } else
#endif
      {
	KernelThreadTaskScheduler *sched = new KernelThreadTaskScheduler(me, *rsrv);
	sched->cfg_max_idle_workers = 3; // keep a few idle threads around
	set_scheduler(sched);
      }
    }

    // slaves run kernel threads because they never context switch
    for(int i = 1; i < num_threads; i++) {
      CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_numa_domain(numa_node);
      params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
      params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(_stack_size);

      std::string name = stringbuilder() << "OMP" << numa_node << " proc " << _me << " (slave " << i << ")";

      CoreReservation *rsrv = new CoreReservation(name, crs, params);
      core_rsrvs.push_back(rsrv);

      // worker threads will be tracked by the threadpool
      ThreadLaunchParameters tlp;
      Thread::create_kernel_thread<ThreadPool, &ThreadPool::worker_entry>(pool,
									  tlp,
									  *rsrv);
    }
  }

  LocalOpenMPProcessor::~LocalOpenMPProcessor(void)
  {
    for(std::vector<CoreReservation *>::const_iterator it = core_rsrvs.begin();
	it != core_rsrvs.end();
	++it)
      delete *it;
    core_rsrvs.clear();
  }

  void LocalOpenMPProcessor::shutdown(void)
  {
    log_omp.info() << "shutting down";
    pool->shutdown();
    delete pool;

    LocalTaskProcessor::shutdown();
  }

  void LocalOpenMPProcessor::execute_task(Processor::TaskFuncID func_id,
					  const ByteArrayRef& task_args)
  {
    // LocalTaskProcessor does most of the work, but make sure we're associated
    //  with the threadpool as a master before we hand off
    pool->associate_as_master();

    LocalTaskProcessor::execute_task(func_id, task_args);
  }


  namespace OpenMP {

    ////////////////////////////////////////////////////////////////////////
    //
    // class OpenMPModule

    OpenMPModule::OpenMPModule(void)
      : Module("openmp")
      , cfg_num_openmp_cpus(0)
      , cfg_num_threads_per_cpu(1)
      , cfg_use_numa(true)
      , cfg_fake_cpukind(false)
      , cfg_stack_size_in_mb(2)
    {
    }
      
    OpenMPModule::~OpenMPModule(void)
    {}

    /*static*/ Module *OpenMPModule::create_module(RuntimeImpl *runtime,
						 std::vector<std::string>& cmdline)
    {
      // create a module to fill in with stuff - we'll delete it if numa is
      //  disabled
      OpenMPModule *m = new OpenMPModule;

      // first order of business - read command line parameters
      {
	CommandLineParser cp;

	cp.add_option_int("-ll:ocpu", m->cfg_num_openmp_cpus)
	  .add_option_int("-ll:othr", m->cfg_num_threads_per_cpu)
	  .add_option_int("-ll:onuma", m->cfg_use_numa)
	  .add_option_int("-ll:ostack", m->cfg_stack_size_in_mb)
	  .add_option_bool("-ll:okindhack", m->cfg_fake_cpukind);
	
	bool ok = cp.parse_command_line(cmdline);
	if(!ok) {
	  log_omp.fatal() << "error reading OpenMP command line parameters";
	  assert(false);
	}
      }

      // if no cpus were requested, there's no point
      if(m->cfg_num_openmp_cpus == 0) {
	log_omp.debug() << "no OpenMP cpus requested";
	delete m;
	return 0;
      }

      // get number/sizes of NUMA nodes -
      //   disable (with a warning) numa binding if support not found
      if(m->cfg_use_numa) {
	std::map<int, NumaNodeCpuInfo> cpuinfo;
	if(numasysif_numa_available() &&
	   numasysif_get_cpu_info(cpuinfo) &&
	   !cpuinfo.empty()) {
	  // filter out any numa domains with insufficient core counts
	  int cores_needed = (m->cfg_num_openmp_cpus *
			      m->cfg_num_threads_per_cpu);
	  for(std::map<int, NumaNodeCpuInfo>::const_iterator it = cpuinfo.begin();
	      it != cpuinfo.end();
	      ++it) {
	    const NumaNodeCpuInfo& ci = it->second;
	    if(ci.cores_available >= cores_needed) {
	      m->active_numa_domains.insert(ci.node_id);
	    } else {
	      log_omp.warning() << "not enough cores in NUMA domain " << ci.node_id << " (" << ci.cores_available << " < " << cores_needed << ")";
	    }
	  }
	} else {
	  log_omp.warning() << "numa support not found (or not working)";
	  m->cfg_use_numa = false;
	}
      }

      // if we don't end up with any active numa domains,
      //  use NUMA_DOMAIN_DONTCARE
      // actually, use the value (-1) since it seems to cause link errors!?
      if(m->active_numa_domains.empty())
	m->active_numa_domains.insert(-1 /*CoreReservationParameters::NUMA_DOMAIN_DONTCARE*/);

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void OpenMPModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void OpenMPModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);

      for(std::set<int>::const_iterator it = active_numa_domains.begin();
	  it != active_numa_domains.end();
	  ++it) {
	int cpu_node = *it;
	for(int i = 0; i < cfg_num_openmp_cpus; i++) {
	  Processor p = runtime->next_local_processor_id();
	  ProcessorImpl *pi = new LocalOpenMPProcessor(p, cpu_node,
						       cfg_num_threads_per_cpu,
						       cfg_fake_cpukind,
						       runtime->core_reservation_set(),
						       cfg_stack_size_in_mb << 20,
						       Config::force_kernel_threads);
	  runtime->add_processor(pi);

	  // FIXME: once the stuff in runtime_impl.cc is removed, remove
	  //  this 'continue' so that we create affinities here
	  if(cfg_fake_cpukind) continue;

	  // create affinities between this processor and system/reg memories
	  // if the memory is one we created, use the kernel-reported distance
	  // to adjust the answer
	  std::vector<MemoryImpl *>& local_mems = runtime->nodes[my_node_id].memories;
	  for(std::vector<MemoryImpl *>::iterator it2 = local_mems.begin();
	      it2 != local_mems.end();
	      ++it2) {
	    Memory::Kind kind = (*it2)->get_kind();
	    if((kind != Memory::SYSTEM_MEM) && (kind != Memory::REGDMA_MEM))
	      continue;

	    Machine::ProcessorMemoryAffinity pma;
	    pma.p = p;
	    pma.m = (*it2)->me;

	    // use the same made-up numbers as in
	    //  runtime_impl.cc
	    if(kind == Memory::SYSTEM_MEM) {
	      pma.bandwidth = 100;  // "large"
	      pma.latency = 5;      // "small"
	    } else {
	      pma.bandwidth = 80;   // "large"
	      pma.latency = 10;     // "small"
	    }
	    
	    runtime->add_proc_mem_affinity(pma);
	  }
	}
      }
    }
    
    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void OpenMPModule::cleanup(void)
    {
      Module::cleanup();
    }

  }; // namespace OpenMP

}; // namespace Realm
