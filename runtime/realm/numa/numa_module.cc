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

#include "realm/numa/numa_module.h"

#include "realm/numa/numasysif.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/proc_impl.h"
#include "realm/threads.h"
#include "realm/runtime_impl.h"
#include "realm/utils.h"

namespace Realm {

  Logger log_numa("numa");

  ////////////////////////////////////////////////////////////////////////
  //
  // class LocalNumaProcessor

  // this is nearly identical to a LocalCPUProcessor, but it asks for its thread(s)
  //  to run on the specified numa domain

  class LocalNumaProcessor : public LocalTaskProcessor {
  public:
    LocalNumaProcessor(Processor _me, int _numa_node,
		       CoreReservationSet& crs, size_t _stack_size,
		       bool _force_kthreads);
    virtual ~LocalNumaProcessor(void);
  protected:
    int numa_node;
    CoreReservation *core_rsrv;
  };

  LocalNumaProcessor::LocalNumaProcessor(Processor _me, int _numa_node,
					 CoreReservationSet& crs,
					 size_t _stack_size,
					 bool _force_kthreads)
    : LocalTaskProcessor(_me, Processor::LOC_PROC)
    , numa_node(_numa_node)
  {
    CoreReservationParameters params;
    params.set_num_cores(1);
    params.set_numa_domain(numa_node);
    params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
    params.set_ldst_usage(params.CORE_USAGE_SHARED);
    params.set_max_stack_size(_stack_size);

    std::string name = stringbuilder() << "NUMA" << numa_node << " proc " << _me;

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
  }

  LocalNumaProcessor::~LocalNumaProcessor(void)
  {
    delete core_rsrv;
  }


  namespace Numa {

    ////////////////////////////////////////////////////////////////////////
    //
    // class NumaModule

    NumaModule::NumaModule(void)
      : Module("numa")
      , cfg_numa_mem_size_in_mb(0)
      , cfg_numa_nocpu_mem_size_in_mb(-1)
      , cfg_num_numa_cpus(0)
      , cfg_pin_memory(false)
      , cfg_stack_size_in_mb(2)
    {
    }
      
    NumaModule::~NumaModule(void)
    {}

    /*static*/ Module *NumaModule::create_module(RuntimeImpl *runtime,
						 std::vector<std::string>& cmdline)
    {
      // create a module to fill in with stuff - we'll delete it if numa is
      //  disabled
      NumaModule *m = new NumaModule;

      // first order of business - read command line parameters
      {
	CommandLineParser cp;

	cp.add_option_int("-ll:nsize", m->cfg_numa_mem_size_in_mb)
	  .add_option_int("-ll:ncsize", m->cfg_numa_nocpu_mem_size_in_mb)
	  .add_option_int("-ll:ncpu", m->cfg_num_numa_cpus)
	  .add_option_bool("-numa:pin", m->cfg_pin_memory);
	
	bool ok = cp.parse_command_line(cmdline);
	if(!ok) {
	  log_numa.fatal() << "error reading NUMA command line parameters";
	  assert(false);
	}
      }

      // if neither NUMA memory nor cpus was requested, there's no point
      if((m->cfg_numa_mem_size_in_mb == 0) &&
	 (m->cfg_numa_nocpu_mem_size_in_mb <= 0) &&
	 (m->cfg_num_numa_cpus == 0)) {
	log_numa.debug() << "no NUMA memory or cpus requested";
	delete m;
	return 0;
      }

      // next step - see if the system supports NUMA allocation/binding
      if(!numasysif_numa_available()) {
	// TODO: warning or fatal error here?
	log_numa.warning() << "numa support not available in system";
	delete m;
	return 0;
      }

      // get number/sizes of NUMA nodes
      std::map<int, NumaNodeMemInfo> meminfo;
      std::map<int, NumaNodeCpuInfo> cpuinfo;
      if(!numasysif_get_mem_info(meminfo) ||
	 !numasysif_get_cpu_info(cpuinfo)) {
	log_numa.fatal() << "failed to get mem/cpu info from system";
	assert(false);
      }

      // some sanity-checks
      for(std::map<int, NumaNodeMemInfo>::const_iterator it = meminfo.begin();
	  it != meminfo.end();
	  ++it) {
	const NumaNodeMemInfo& mi = it->second;
	log_numa.info() << "NUMA memory node " << mi.node_id << ": " << (mi.bytes_available >> 20) << " MB";

	size_t mem_size = (m->cfg_numa_mem_size_in_mb << 20);
	if(m->cfg_numa_nocpu_mem_size_in_mb >= 0) {
	  // use this value instead if there are no cpus in this domain
	  if(cpuinfo.count(mi.node_id) == 0)
	    mem_size = (m->cfg_numa_nocpu_mem_size_in_mb << 20);
	}

	// skip domain silently if no memory is requested
	if(mem_size == 0)
	  continue;

	if(mi.bytes_available >= mem_size) {
	  m->numa_mem_bases[mi.node_id] = 0;
	  m->numa_mem_sizes[mi.node_id] = mem_size;
	} else {
	  // TODO: fatal error?
	  log_numa.warning() << "insufficient memory in NUMA node " << mi.node_id << " (" << mem_size << " > " << mi.bytes_available << " bytes) - skipping allocation";
	}
      }
      for(std::map<int, NumaNodeCpuInfo>::const_iterator it = cpuinfo.begin();
	  it != cpuinfo.end();
	  ++it) {
	const NumaNodeCpuInfo& ci = it->second;
	log_numa.info() << "NUMA cpu node " << ci.node_id << ": " << ci.cores_available << " cores";
	if(ci.cores_available >= m->cfg_num_numa_cpus) {
	  m->numa_cpu_counts[ci.node_id] = m->cfg_num_numa_cpus;
	} else {
	  // TODO: fatal error?
	  log_numa.warning() << "insufficient cores in NUMA node " << ci.node_id << " - core assignment will fail";
	  m->numa_cpu_counts[ci.node_id] = m->cfg_num_numa_cpus;
	}
      }

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void NumaModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);

      // memory allocations are performed here
      for(std::map<int, void *>::iterator it = numa_mem_bases.begin();
	  it != numa_mem_bases.end();
	  ++it) {
	size_t mem_size = numa_mem_sizes[it->first];
	assert(mem_size > 0);
	void *base = numasysif_alloc_mem(it->first,
					 mem_size,
					 cfg_pin_memory);
	if(!base) {
	  log_numa.fatal() << "allocation of " << mem_size << " bytes in NUMA node " << it->first << " failed!";
	  assert(false);
	}
	it->second = base;
      }
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void NumaModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);

      for(std::map<int, void *>::iterator it = numa_mem_bases.begin();
	  it != numa_mem_bases.end();
	  ++it) {
	int mem_node = it->first;
	void *base_ptr = it->second;
	size_t mem_size = numa_mem_sizes[it->first];
	assert(mem_size > 0);

	Memory m = runtime->next_local_memory_id();
	LocalCPUMemory *numamem = new LocalCPUMemory(m,
						     mem_size,
						     base_ptr,
						     false /*!registered*/);
	runtime->add_memory(numamem);
	memories[mem_node] = numamem;
      }
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void NumaModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);

      for(std::map<int, int>::const_iterator it = numa_cpu_counts.begin();
	  it != numa_cpu_counts.end();
	  ++it) {
	int cpu_node = it->first;
	for(int i = 0; i < it->second; i++) {
	  Processor p = runtime->next_local_processor_id();
	  ProcessorImpl *pi = new LocalNumaProcessor(p, it->first,
						     runtime->core_reservation_set(),
						     cfg_stack_size_in_mb << 20,
						     Config::force_kernel_threads);
	  runtime->add_processor(pi);

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

	    int mem_node = -1;
	    for(std::map<int, MemoryImpl *>::const_iterator it3 = memories.begin();
		it3 != memories.end();
		++it3)
	      if(it3->second == *it2) {
		mem_node = it3->first;
		break;
	      }

	    if(mem_node == -1) {
	      // not one of our memories - use the same made-up numbers as in
	      //  runtime_impl.cc
	      if(kind == Memory::SYSTEM_MEM) {
		pma.bandwidth = 100;  // "large"
		pma.latency = 5;      // "small"
	      } else {
		pma.bandwidth = 80;   // "large"
		pma.latency = 10;     // "small"
	      }
	    } else {
	      int d = numasysif_get_distance(cpu_node, mem_node);
	      if(d >= 0) {
		pma.bandwidth = 150 - d;
		pma.latency = d / 10;     // Linux uses a cost of ~10/hop
	      } else {
		// same as random sysmem
		pma.bandwidth = 100;
		pma.latency = 5;
	      }
	    }
	    
	    runtime->add_proc_mem_affinity(pma);
	  }
	}
      }
    }
    
    // create any DMA channels provided by the module (default == do nothing)
    void NumaModule::create_dma_channels(RuntimeImpl *runtime)
    {
      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void NumaModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void NumaModule::cleanup(void)
    {
      Module::cleanup();

      // free our allocations here
      for(std::map<int, void *>::iterator it = numa_mem_bases.begin();
	  it != numa_mem_bases.end();
	  ++it) {
	size_t mem_size = numa_mem_sizes[it->first];
	assert(mem_size > 0);
	bool ok = numasysif_free_mem(it->first, it->second, mem_size);
	if(!ok)
	  log_numa.error() << "failed to free memory in NUMA node " << it->first << ": ptr=" << it->second;
      }
    }

  }; // namespace Numa

}; // namespace Realm
