#include "realm/accelerator/accelerator_module.h"


#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/threads.h"
#include "realm/utils.h"

// each task access by include header file where the namespace is declared
namespace XRTContext {
  // define extern xrt_device
  thread_local XRTDevice<HW_EXP_SIZE, HW_MANT_SIZE> *xrt_device = 0;
}

namespace Realm {
  namespace Accelerator {

    Logger log_accel("accel");

    AcceleratorModule::AcceleratorModule() : Module("accelerator"), cfg_num_accelerators_(0) {
    }

    AcceleratorModule::~AcceleratorModule(void) {}

    Module *AcceleratorModule::create_module(RuntimeImpl *runtime, std::vector<std::string>& cmdline) {
      AcceleratorModule *m = new AcceleratorModule;
      log_accel.info() << "use accelerator";
      Realm::CommandLineParser cp;
      cp.add_option_string("-accel:fwbin", m->cfg_fwbin_path_);
      cp.add_option_int("-ll:num_accelerators", m->cfg_num_accelerators_);

      bool ok = cp.parse_command_line(cmdline);
      if (!ok) {
        log_accel.error() << "error reading accelerator parameters";
        exit(1);
      }

      for (int i = 0; i < m->cfg_num_accelerators_; i++) {
        // template arguments must be known at compile time and const
        // TODO: add support for non-xilinx fpgas
        XRTDevice<HW_EXP_SIZE, HW_MANT_SIZE> *xrt = new XRTDevice<HW_EXP_SIZE, HW_MANT_SIZE>(m->cfg_fwbin_path_);
        m->xrt_devices_.push_back(xrt);
      }

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void AcceleratorModule::initialize(RuntimeImpl *runtime) {
      Module::initialize(runtime);
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void AcceleratorModule::create_memories(RuntimeImpl *runtime) {
      Module::create_memories(runtime);
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void AcceleratorModule::create_processors(RuntimeImpl *runtime) {
      Module::create_processors(runtime); 
      // 1 : 1 mapping processor to device
      for (int i = 0; i < xrt_devices_.size(); i++) {
        Processor p = runtime->next_local_processor_id();
        AcceleratorProcessor *proc = new AcceleratorProcessor(xrt_devices_[i], p, runtime->core_reservation_set());
        procs_.push_back(proc);
        runtime->add_processor(proc);

        // create mem affinities to add a proc to machine model
        // create affinities between this processor and system/reg memories
        // if the memory is one we created, use the kernel-reported distance
        // to adjust the answer
        std::vector<MemoryImpl *>& local_mems = runtime->nodes[Network::my_node_id].memories;
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

    // create any DMA channels provided by the module (default == do nothing)
    void AcceleratorModule::create_dma_channels(RuntimeImpl *runtime) {
      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void AcceleratorModule::create_code_translators(RuntimeImpl *runtime) {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void AcceleratorModule::cleanup(void) {
      for (std::vector<XRTDevice<HW_EXP_SIZE, HW_MANT_SIZE> *>::iterator it = xrt_devices_.begin(); it != xrt_devices_.end(); it++)
        delete *it;
      xrt_devices_.clear();
    }

    template <typename T>
    class AcceleratorTaskScheduler : public T {
      public:
        AcceleratorTaskScheduler(Processor proc, Realm::CoreReservation& core_rsrv, AcceleratorProcessor *accelerator_proc);
        virtual ~AcceleratorTaskScheduler(void);
      protected:
        virtual bool execute_task(Task *task);
        virtual void execute_internal_task(InternalTask *task);
        AcceleratorProcessor *accel_proc_;
    };

    template <typename T>
    AcceleratorTaskScheduler<T>::AcceleratorTaskScheduler(Processor proc,
                                                          Realm::CoreReservation& core_rsrv,
                                                          AcceleratorProcessor *accel_proc) : T(proc, core_rsrv), accel_proc_(accel_proc) {
    }

    template <typename T>
      AcceleratorTaskScheduler<T>::~AcceleratorTaskScheduler(void) {
    }

    template <typename T>
    bool AcceleratorTaskScheduler<T>::execute_task(Task *task) {
      // add device to thread's xrt context
      XRTContext::xrt_device = accel_proc_->xrt_device_;
      bool ok = T::execute_task(task);
      return ok;
    }

    template <typename T>
    void AcceleratorTaskScheduler<T>::execute_internal_task(InternalTask *task) {
      // add device to thread's xrt context
      XRTContext::xrt_device = accel_proc_->xrt_device_;
      T::execute_internal_task(task);
    }

    AcceleratorProcessor::AcceleratorProcessor(XRTDevice<HW_EXP_SIZE, HW_MANT_SIZE> *xrt, Processor me, Realm::CoreReservationSet& crs)
    : LocalTaskProcessor(me, Processor::ACCEL_PROC)
    {
      xrt_device_ = xrt;

      Realm::CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_SHARED);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(2 << 20);
      std::string name = stringbuilder() << "Accel proc " << me;
      core_rsrv_ = new Realm::CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS
      UserThreadTaskScheduler *sched = new AcceleratorTaskScheduler<UserThreadTaskScheduler>(me, *core_rsrv_, this);
#else
      KernelThreadTaskScheduler *sched = new AcceleratorTaskScheduler<KernelThreadTaskScheduler>(me, *core_rsrv_, this);
#endif
      set_scheduler(sched);
    }

    AcceleratorProcessor::~AcceleratorProcessor(void) {
      delete core_rsrv_;
    }

  }; // namespace Accelerator
}; // namespace Realm

