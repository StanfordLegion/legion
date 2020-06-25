#ifndef REALM_ACCELERATOR_MODULE_H
#define REALM_ACCELERATOR_MODULE_H

#include "hls/cpfp_conv.h" // class XRTdevice

#include "realm/module.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/runtime_impl.h"

namespace Realm {
  namespace Accelerator {

    class AcceleratorModule;
    class AcceleratorProcessor;

    class AcceleratorModule : public Module {
      protected:
        AcceleratorModule(void);

      public:
        virtual ~AcceleratorModule(void);

        static Module *create_module(RuntimeImpl *runtime, std::vector<std::string>& cmdline);

        // do any general initialization - this is called after all configuration is
        //  complete
        virtual void initialize(RuntimeImpl *runtime);

        // create any memories provided by this module (default == do nothing)
        //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
        virtual void create_memories(RuntimeImpl *runtime);

        // create any processors provided by the module (default == do nothing)
        //  (each new ProcessorImpl should use a Processor from
        //   RuntimeImpl::next_local_processor_id)
        virtual void create_processors(RuntimeImpl *runtime);

        // create any DMA channels provided by the module (default == do nothing)
        virtual void create_dma_channels(RuntimeImpl *runtime);

        // create any code translators provided by the module (default == do nothing)
        virtual void create_code_translators(RuntimeImpl *runtime);

        // clean up any common resources created by the module - this will be called
        //  after all memories/processors/etc. have been shut down and destroyed
        virtual void cleanup(void);

      public:
        unsigned cfg_num_accelerators_;
        std::string cfg_fwbin_path_;

      protected:
  	std::vector<AcceleratorProcessor *> procs_;
        std::vector<XRTDevice<HW_EXP_SIZE, HW_MANT_SIZE> *> xrt_devices_;
    };

    REGISTER_REALM_MODULE(AcceleratorModule);

    class AcceleratorProcessor : public LocalTaskProcessor {
      public:
        AcceleratorProcessor(XRTDevice<HW_EXP_SIZE, HW_MANT_SIZE> *xrt, Processor me, Realm::CoreReservationSet& crs);
        virtual ~AcceleratorProcessor(void);

        XRTDevice<HW_EXP_SIZE, HW_MANT_SIZE> *xrt_device_;
      
      protected:
        Realm::CoreReservation *core_rsrv_;
    };

  }; // namespace Accelerator
}; // namespace Realm

#endif

