/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include "realm/openmp/openmp_internal.h"

#ifndef REALM_OPENMP_SYSTEM_RUNTIME
#include "realm/openmp/openmp_threadpool.h"
#endif

#include "realm/numa/numasysif.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/proc_impl.h"
#include "realm/threads.h"
#include "realm/runtime_impl.h"
#include "realm/utils.h"

#ifdef REALM_OPENMP_SYSTEM_RUNTIME
#include <omp.h>
#endif

namespace Realm {

  Logger log_omp("openmp");

#ifndef REALM_OPENMP_SYSTEM_RUNTIME
  // defined in openmp_api.cc - refer to it to force linkage of that file
  extern void openmp_api_force_linkage(void);
#endif


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalOpenMPProcessor

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
    , ctxmgr(this)
  {
    // master runs in a user threads if possible
    {
      CoreReservationParameters params;
#ifdef REALM_OPENMP_SYSTEM_RUNTIME
      params.set_num_cores(num_threads); // system omp runtime will use these
#else
      params.set_num_cores(1);
#endif
      params.set_numa_domain(numa_node);
      params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
      params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(_stack_size);

      std::string name = stringbuilder() << "OMP" << numa_node << " proc " << _me << " (master)";

      core_rsrv = new CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS
      if(!_force_kthreads) {
	UserThreadTaskScheduler *sched = new UserThreadTaskScheduler(me, *core_rsrv);
	// no config settings we want to tweak yet
	set_scheduler(sched);
      } else
#endif
      {
	KernelThreadTaskScheduler *sched = new KernelThreadTaskScheduler(me, *core_rsrv);
	sched->cfg_max_idle_workers = 3; // keep a few idle threads around
	set_scheduler(sched);
      }
      sched->add_task_context(&ctxmgr);
    }

#ifndef REALM_OPENMP_SYSTEM_RUNTIME
    pool = new ThreadPool(me,
                          num_threads - 1,
			  stringbuilder() << "OMP" << numa_node << " proc " << _me,
			  numa_node, _stack_size, crs);

    // eagerly spin up worker threads
    pool->start_worker_threads();
#endif
  }

  LocalOpenMPProcessor::~LocalOpenMPProcessor(void)
  {
    delete core_rsrv;
  }

  void LocalOpenMPProcessor::shutdown(void)
  {
    log_omp.info() << "shutting down";
#ifndef REALM_OPENMP_SYSTEM_RUNTIME
    pool->stop_worker_threads();
    delete pool;
#endif

    LocalTaskProcessor::shutdown();
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalOpenMPProcessor::OpenMPContextManager

  LocalOpenMPProcessor::OpenMPContextManager::OpenMPContextManager(LocalOpenMPProcessor *_proc)
    : proc(_proc)
  {}

  void *LocalOpenMPProcessor::OpenMPContextManager::create_context(Task *task) const
  {
#ifdef REALM_OPENMP_SYSTEM_RUNTIME
    // this must be set on the right thread
    omp_set_num_threads(proc->num_threads);

    // make sure all of our workers know who we are
    #pragma omp parallel
    {
      ThreadLocal::current_processor = proc->me;
    }
#else
    proc->pool->associate_as_master();
#endif

    // we don't need to remember anything
    return 0;
  }

  void LocalOpenMPProcessor::OpenMPContextManager::destroy_context(Task *task, void *context) const
  {
    // nothing to clean up
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
      , cfg_stack_size(2 << 20)
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

#ifndef REALM_OPENMP_SYSTEM_RUNTIME
      openmp_api_force_linkage();
#endif

      // first order of business - read command line parameters
      {
	CommandLineParser cp;

	cp.add_option_int("-ll:ocpu", m->cfg_num_openmp_cpus)
	  .add_option_int("-ll:othr", m->cfg_num_threads_per_cpu)
	  .add_option_int("-ll:onuma", m->cfg_use_numa)
	  .add_option_int_units("-ll:ostack", m->cfg_stack_size, 'm')
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

#ifdef REALM_OPENMP_SYSTEM_RUNTIME
      if(m->cfg_num_openmp_cpus > 1) {
	log_omp.fatal() << "system omp runtime limited to 1 proc - " << m->cfg_num_openmp_cpus << " requested";
        abort();
      }
#endif

      // get number/sizes of NUMA nodes -
      //   disable (with a warning) numa binding if support not found
      if(m->cfg_use_numa) {
	std::map<int, NumaNodeCpuInfo> cpuinfo;
	if(numasysif_numa_available() &&
	   numasysif_get_cpu_info(cpuinfo) &&
	   !cpuinfo.empty()) {
          // Figure out how many OpenMP processors we need per NUMA domain
          int openmp_cpus_per_numa_node = 
            (m->cfg_num_openmp_cpus + cpuinfo.size() - 1) / cpuinfo.size();
	  int cores_needed = (openmp_cpus_per_numa_node *
			      m->cfg_num_threads_per_cpu);
	  // filter out any numa domains with insufficient core counts
	  for(std::map<int, NumaNodeCpuInfo>::const_iterator it = cpuinfo.begin();
	      it != cpuinfo.end();
	      ++it) {
	    const NumaNodeCpuInfo& ci = it->second;
	    if(ci.cores_available >= cores_needed) {
	      m->active_numa_domains.push_back(ci.node_id);
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
	m->active_numa_domains.push_back(-1 /*CoreReservationParameters::NUMA_DOMAIN_DONTCARE*/);

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

      assert(!active_numa_domains.empty());
      for(int i = 0; i < cfg_num_openmp_cpus; i++) {
        int cpu_node = active_numa_domains[i % active_numa_domains.size()];
        Processor p = runtime->next_local_processor_id();
        ProcessorImpl *pi = new LocalOpenMPProcessor(p, cpu_node,
                                                     cfg_num_threads_per_cpu,
                                                     cfg_fake_cpukind,
                                                     runtime->core_reservation_set(),
                                                     cfg_stack_size,
                                                     Config::force_kernel_threads);
        runtime->add_processor(pi);

        // FIXME: once the stuff in runtime_impl.cc is removed, remove
        //  this 'continue' so that we create affinities here
        if(cfg_fake_cpukind) continue;

        // create affinities between this processor and system/reg memories
        // if the memory is one we created, use the kernel-reported distance
        // to adjust the answer
        std::vector<MemoryImpl *>& local_mems = runtime->nodes[Network::my_node_id].memories;
        for(std::vector<MemoryImpl *>::iterator it2 = local_mems.begin();
            it2 != local_mems.end();
            ++it2) {
          Memory::Kind kind = (*it2)->get_kind();
          if((kind != Memory::SYSTEM_MEM) && (kind != Memory::REGDMA_MEM) &&
             (kind != Memory::SOCKET_MEM) && (kind != Memory::Z_COPY_MEM))
            continue;

          Machine::ProcessorMemoryAffinity pma;
          pma.p = p;
          pma.m = (*it2)->me;

          // use the same made-up numbers as in
          //  runtime_impl.cc
          if(kind == Memory::SYSTEM_MEM) {
            pma.bandwidth = 100;  // "large"
            pma.latency = 5;      // "small"
          } else if (kind == Memory::Z_COPY_MEM) {
            pma.bandwidth = 40; // "large"
            pma.latency = 3; // "small"
          } else if (kind == Memory::REGDMA_MEM) {
            pma.bandwidth = 80;   // "large"
            pma.latency = 10;     // "small"
          } else {
            // This is a numa domain, see if it is the same as ours or not
            if (cfg_use_numa) {
              // Figure out which numa node the memory is in
              LocalCPUMemory *cpu_mem = static_cast<LocalCPUMemory*>(*it2);
              int mem_node = cpu_mem->numa_node;
              assert(mem_node >= 0);
              // We know our numa node
              int distance = numasysif_get_distance(cpu_node, mem_node);
              if (distance >= 0) {
                pma.bandwidth = 150 - distance;
                pma.latency = distance / 10;     // Linux uses a cost of ~10/hop
              } else {
                // same as random sysmem
                pma.bandwidth = 100;
                pma.latency = 5;
              }
            } else {
              // NUMA not available so use system memory settings
              pma.bandwidth = 100; // "large"
              pma.latency = 5; // "small"
            }
          }
          
          runtime->add_proc_mem_affinity(pma);
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
