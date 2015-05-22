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

local ef = terralib.externfunction

local RuntimeAPI = terralib.includec("cuda_runtime.h")
RuntimeAPI.__cudaRegisterCudaBinary =
  ef("__cudaRegisterCudaBinary", {&opaque, uint64} -> &&opaque)
RuntimeAPI.__cudaRegisterFunction =
  ef("__cudaRegisterFunction",
     {&opaque, &int8, &int8, &int8, int,
      &RuntimeAPI.uint3, &RuntimeAPI.uint3,
      &RuntimeAPI.dim3, &RuntimeAPI.dim3, &int} -> {})

local C = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]]

local terra assert(x : bool, message : rawstring)
  if not x then
    var stderr = C.fdopen(2, "w")
    C.fprintf(stderr, "assertion failed: %s\n", message)
    -- Just because it's stderr doesn't mean it's unbuffered...
    C.fflush(stderr)
    C.abort()
  end
end

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
local DeviceAPI = {
  cuInit = ef("cuInit", {uint32} -> uint32);
  cuCtxGetCurrent = ef("cuCtxGetCurrent", {&&CUctx_st} -> uint32);
  cuDeviceGet = ef("cuDeviceGet",{&int32,int32} -> uint32);
  cuCtxCreate_v2 = ef("cuCtxCreate_v2",{&&CUctx_st,uint32,int32} -> uint32);
  cuCtxGetDevice = ef("cuCtxGetDevice", {&CUdevice} -> uint32);
  cuLinkCreate_v2 = ef("cuLinkCreate_v2",
    {uint32,&uint32,&&opaque,&&CUlinkState_st} -> uint32);
  cuLinkAddData_v2 = ef("cuLinkAddData_v2",
    {&CUlinkState_st,uint32,&opaque,uint64,&int8,uint32,&uint32,&&opaque} -> uint32);
  cuLinkComplete = ef("cuLinkComplete",
    {&CUlinkState_st,&&opaque,&uint64} -> uint32);
  cuLinkDestroy = ef("cuLinkDestroy",
    {&CUlinkState_st} -> uint32);
}

local terra compile_ptx(ptxc : rawstring, ptxSize : uint32) : &&opaque
  DeviceAPI.cuInit(0)
  var cx : &CUctx_st
  DeviceAPI.cuCtxGetCurrent(&cx)
  if cx == nil then
    C.printf("new cuda context created\n")
    var d : int32
    DeviceAPI.cuDeviceGet(&d, 0)
    DeviceAPI.cuCtxCreate_v2(&cx, 0, d)
  end

  var linkState : &CUlinkState_st
  var cubin : &opaque
  var cubinSize : uint64
  var options = arrayof(CUjit_option, CU_JIT_TARGET, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
  var error_str : rawstring
  var error_sz : uint64
  var version = [cudalib.localversion()]
  var option_values = arrayof([&opaque], [&opaque](version), error_str, [&opaque](error_sz))
  var r = DeviceAPI.cuLinkCreate_v2(3, options, option_values, &linkState)
  assert(r == 0, "CUDA error in creating linker")
  r = DeviceAPI.cuLinkAddData_v2(linkState, CU_JIT_INPUT_PTX, ptxc, ptxSize, nil, 0, nil, nil)
  assert(r == 0, "CUDA error in adding PTX")
  r = DeviceAPI.cuLinkComplete(linkState, &cubin, &cubinSize)
  assert(r == 0, "CUDA error in linking")

  var handle = RuntimeAPI.__cudaRegisterCudaBinary(cubin, cubinSize)

  r = DeviceAPI.cuLinkDestroy(linkState)
  assert(r == 0, "CUDA error in destroying linker")

  return handle
end

local terra register_function(handle : &&opaque, id : int, name : &int8)
  RuntimeAPI.__cudaRegisterFunction(handle, [&int8](id), name,
                                    nil, 0, nil, nil, nil, nil, nil)
end

function cudahelper.jit_compile_kernels_and_register(kernels)
  local module = {}
  for k, v in pairs(kernels) do
    module[v.name] = v.kernel
  end
  local ptx = cudalib.toptx(module, nil, cudalib.localversion())
  local ptxc = terralib.constant(ptx)
  local ptxSize = ptx:len() + 1
  local handle = compile_ptx(ptx, ptxSize)

  for k, v in pairs(kernels) do
    register_function(handle, k, v.name)
  end
end

function cudahelper.codegen_kernel_call(kernel_id, count, args)
  local setupArguments = terralib.newlist()

  local offset = 0
  for _, arg in ipairs(args) do
    local size = terralib.sizeof(arg.type)
    setupArguments:insert(quote
      RuntimeAPI.cudaSetupArgument(&[arg], size, offset)
    end)
    offset = offset + 8
  end

  return quote
    var grid : RuntimeAPI.dim3, block : RuntimeAPI.dim3
    var threadsPerBlock : uint = 256 -- hard-coded for the moment
    var numBlocks : uint = ([count] + (threadsPerBlock - 1)) / threadsPerBlock
    grid.x, grid.y, grid.z = numBlocks, 1, 1
    block.x, block.y, block.z = threadsPerBlock, 1, 1
    RuntimeAPI.cudaConfigureCall(grid, block, 0, nil)
    [setupArguments];
    RuntimeAPI.cudaLaunch([&int8](kernel_id))
  end
end

return cudahelper
