#include "realm/fpga/fpga_module.h"
#include "realm/fpga/fpga_utils.h"
#include "realm/logging.h"

namespace Realm
{
    namespace FPGA
    {
        extern Logger log_fpga;
    }
}

extern "C"
{
    using namespace Realm;
    using namespace Realm::FPGA;

    REALM_PUBLIC_API cl::Device FPGAGetCurrentDevice(void)
    {
        FPGAProcessor *p = FPGAProcessor::get_current_fpga_proc();
        cl::Device ret = p->fpga_device->device;
        log_fpga.info() << "FPGAGetCurrentDevice()";
        return ret;
    }

    REALM_PUBLIC_API cl::Context FPGAGetCurrentContext(void)
    {
        FPGAProcessor *p = FPGAProcessor::get_current_fpga_proc();
        cl::Context ret = p->fpga_device->context;
        log_fpga.info() << "FPGAGetCurrentContext()";
        return ret;
    }

    REALM_PUBLIC_API cl::Program FPGAGetCurrentProgram(void)
    {
        FPGAProcessor *p = FPGAProcessor::get_current_fpga_proc();
        cl::Program ret = p->fpga_device->program;
        log_fpga.info() << "FPGAGetCurrentProgram()";
        return ret;
    }

    REALM_PUBLIC_API cl::Buffer FPGAGetCurrentBuffer(void)
    {
        FPGAProcessor *p = FPGAProcessor::get_current_fpga_proc();
        cl::Buffer ret = p->fpga_device->buff;
        log_fpga.info() << "FPGAGetCurrentBuffer()";
        return ret;
    }

    REALM_PUBLIC_API cl::CommandQueue FPGAGetCurrentCommandQueue(void)
    {
        FPGAProcessor *p = FPGAProcessor::get_current_fpga_proc();
        cl::CommandQueue ret = p->fpga_device->command_queue;
        log_fpga.info() << "FPGAGetCurrentCommandQueue()";
        return ret;
    }

    REALM_PUBLIC_API void *FPGAGetBasePtrSys(void)
    {
        FPGAProcessor *p = FPGAProcessor::get_current_fpga_proc();
        void *ret = p->fpga_device->fpga_mem->base_ptr_sys;
        return ret;
    }
}
