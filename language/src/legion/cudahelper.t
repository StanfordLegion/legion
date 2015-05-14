-- Copyright 2015 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

local cudahelper = {}

if not terralib.cudacompile then return cudahelper end

local RuntimeAPI = terralib.includec("cuda_runtime.h")

local struct CUctx_st
local struct CUmod_st
local struct CUlinkState_st
local struct CUfunc_st
local CUdevice = int32
local CUjit_option = uint32
local CU_JIT_ERROR_LOG_BUFFER = 5
local CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
local CU_JIT_INPUT_PTX = 1
local CU_JIT_TARGET = 9
local ef = terralib.externfunction
local DeviceAPI = {
  cuInit = ef("cuInit", {uint32} -> uint32);
  cuCtxGetCurrent = ef("cuCtxGetCurrent", {&&CUctx_st} -> uint32);
  cuCtxGetDevice = ef("cuCtxGetDevice", {&CUdevice} -> uint32);
  cuDeviceComputeCapability = ef("cuDeviceComputeCapability",
    {&int32,&int32,int32} -> uint32);
  cuLinkCreate_v2 = ef("cuLinkCreate_v2",
    {uint32,&uint32,&&opaque,&&CUlinkState_st} -> uint32);
  cuLinkAddData_v2 = ef("cuLinkAddData_v2",
    {&CUlinkState_st,uint32,&opaque,uint64,&int8,uint32,&uint32,&&opaque} -> uint32);
  cuLinkComplete = ef("cuLinkComplete",
    {&CUlinkState_st,&&opaque,&uint64} -> uint32);
  cuLinkDestroy = ef("cuLinkDestroy",
    {&CUlinkState_st} -> uint32);
  cuModuleLoadData = ef("cuModuleLoadData",
    {&&CUmod_st,&opaque} -> uint32);
  cuModuleGetFunction = ef("cuModuleGetFunction",
    {&&CUfunc_st,&CUmod_st,&int8} -> uint32);
}

function cudahelper.codegen_ptx_load(kernel)
  local ptx = cudalib.toptx({ kernel = kernel }, nil, cudalib.localversion())
  local ptxc = terralib.constant(ptx)
  local ptxSize = ptx:len() + 1
  local module_ptr = terralib.newsymbol(&CUmod_st, "cudaM")
  local module_name = terralib.constant("kernel")

  local load_ptx = quote
    var linkState : &CUlinkState_st
    var cubin : &opaque
    var cubinSize : uint64
    var options = arrayof(CUjit_option, CU_JIT_TARGET, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
    var error_str : rawstring
    var error_sz : uint64
    var version = [cudalib.localversion()]
    var option_values = arrayof([&opaque], [&opaque](version), error_str, [&opaque](error_sz))
    DeviceAPI.cuLinkCreate_v2(3, options, option_values, &linkState)
    DeviceAPI.cuLinkAddData_v2(linkState, CU_JIT_INPUT_PTX, ptxc, ptxSize, nil, 0, nil, nil)
    DeviceAPI.cuLinkComplete(linkState, &cubin, &cubinSize)
    DeviceAPI.cuLinkDestroy(linkState)

    var [module_ptr]
    DeviceAPI.cuModuleLoadData(&[module_ptr], cubin)
  end

  return load_ptx, module_ptr, module_name
end

function cudahelper.codegen_get_function(module_ptr, module_name)
  local fn = terralib.newsymbol(&CUfunc_st, "fn")
  local get_function = quote
    var [fn]
    DeviceAPI.cuModuleGetFunction(&[fn], [module_ptr], [module_name])
  end
  return get_function, fn
end

function cudahelper.codegen_kernel_call(fn, count, args)
  local setupArguments = terralib.newlist()

  local offset = 0
  for _, arg in ipairs(args) do
    local size = terralib.sizeof(arg.type)
    setupArguments:insert(quote
      RuntimeAPI.cudaSetupArgument(&[arg], size, offset)
    end)
    offset = offset + size
  end

  return quote
    var grid : RuntimeAPI.dim3, block : RuntimeAPI.dim3
    grid.x, grid.y, grid.z = 1, 1, 1
    block.x, block.y, block.z = [count], 1, 1
    RuntimeAPI.cudaConfigureCall(grid, block, 0, nil)
    [setupArguments];
    RuntimeAPI.cudaLaunch([fn])
  end
end

return cudahelper
