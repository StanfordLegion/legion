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

#include "realm/procset/procset_module.h"

#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/proc_impl.h"
#include "realm/threads.h"
#include "realm/runtime_impl.h"
#include "realm/utils.h"

namespace Realm {

   Logger log_procset("procset");

   ////////////////////////////////////////////////////////////////////////
   //
   // class LocalProcessorSet
   //
   class LocalProcessorSet : public LocalTaskProcessor {
   public:
     LocalProcessorSet(Processor _me, CoreReservationSet& crs,
		       size_t _stack_size, int _num_cores,
		       bool _force_kthreads);
     virtual ~LocalProcessorSet(void);
   protected:
     CoreReservation *core_rsrv;
   };


   LocalProcessorSet::LocalProcessorSet(Processor _me, CoreReservationSet& crs,
					size_t _stack_size, int _num_cores,
					bool _force_kthreads)
     : LocalTaskProcessor(_me, Processor::PROC_SET, _num_cores)

   {
     CoreReservationParameters params;
     params.set_num_cores(_num_cores);
     params.set_alu_usage(params.CORE_USAGE_EXCLUSIVE);
     params.set_fpu_usage(params.CORE_USAGE_EXCLUSIVE);
     params.set_ldst_usage(params.CORE_USAGE_SHARED);
     params.set_max_stack_size(_stack_size);

     std::string name = stringbuilder() << "proc set " << _me;

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

   LocalProcessorSet::~LocalProcessorSet(void)
   {
     delete core_rsrv;
   }


  namespace ProcSet {

    ////////////////////////////////////////////////////////////////////////
    //
    // class ProcSetModule

    ProcSetModule::ProcSetModule(void)
      : Module("procset")
      , cfg_num_mp_threads(0)
      , cfg_num_mp_procs(0)
      , cfg_num_mp_cpus(0)
      , cfg_stack_size(2 << 20)
    {
    }
      
    ProcSetModule::~ProcSetModule(void)
    {}

    /*static*/ Module *ProcSetModule::create_module(RuntimeImpl *runtime,
						 std::vector<std::string>& cmdline)
    {
      // create a module to fill in with stuff 
      ProcSetModule *m = new ProcSetModule;

      // read command line parameters
	  CommandLineParser cp;

      cp.add_option_int("-ll:mp_threads", m->cfg_num_mp_threads)
        .add_option_int("-ll:mp_nodes", m->cfg_num_mp_procs)
        .add_option_int("-ll:mp_cpu", m->cfg_num_mp_cpus);

      bool ok = cp.parse_command_line(cmdline);
      if(!ok) {
	log_procset.fatal() << "error reading ProcSet command line parameters";
	assert(false);
      }
      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void ProcSetModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void ProcSetModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void ProcSetModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);
      // for simplicity don't allow more that one procset per node for now
      if (cfg_num_mp_procs > (Network::max_node_id + 1)) {
	    log_procset.fatal() << "error num_mp_procs > number of nodes";
	    assert(false);
      }
      if (cfg_num_mp_threads) {
        // if num_mp_procs is not set then assume one procset on every node
        if (cfg_num_mp_procs == 0 || Network::my_node_id < cfg_num_mp_procs) { 
          Processor p = runtime->next_local_processor_id();
          ProcessorImpl *pi = new LocalProcessorSet(p, runtime->core_reservation_set(),
						    cfg_stack_size, cfg_num_mp_threads,
						    Config::force_kernel_threads);
          runtime->add_processor(pi);
        // if there are not procSets on all nodes and cfg_num_mp_cpus is set
        // then add additional LocalCPUProcessors on these nodes
        } else if (cfg_num_mp_cpus) {
          for (int i = 0; i < cfg_num_mp_cpus; i++) {
            Processor p = runtime->next_local_processor_id();
            ProcessorImpl *pi = new LocalCPUProcessor(p, runtime->core_reservation_set(),
						      cfg_stack_size,
						      Config::force_kernel_threads, 0, 0);
            runtime->add_processor(pi);
          }
        }      
      }
    }
    
    // create any DMA channels provided by the module (default == do nothing)
    void ProcSetModule::create_dma_channels(RuntimeImpl *runtime)
    {
      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void ProcSetModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void ProcSetModule::cleanup(void)
    {
      Module::cleanup();
    }

  }; // namespace ProcSet

}; // namespace Realm
