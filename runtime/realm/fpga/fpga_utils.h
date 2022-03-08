#ifndef FPGA_UTILS_H
#define FPGA_UTILS_H

#include "xcl2.hpp"
extern "C"
{
    cl::Device FPGAGetCurrentDevice(void);
    cl::Context FPGAGetCurrentContext(void);
    cl::Program FPGAGetCurrentProgram(void);
    cl::Buffer FPGAGetCurrentBuffer(void);
    cl::CommandQueue FPGAGetCurrentCommandQueue(void);
    void *FPGAGetBasePtrSys(void);
}
#endif // FPGA_UTILS_H
