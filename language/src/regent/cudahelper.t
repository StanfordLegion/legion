-- Copyright 2017 Stanford University
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

local config = require("regent/config")
local report = require("common/report")

local cudahelper = {}
cudahelper.check_cuda_available = function() return false end

if not config.args()["cuda"] or not terralib.cudacompile then
  return cudahelper
end

-- copied and modified from cudalib.lua in Terra interpreter

local ffi = require('ffi')

local cudapaths = { OSX = "/usr/local/cuda/lib/libcuda.dylib";
                    Linux =  "libcuda.so";
                    Windows = "nvcuda.dll"; }

local cudaruntimelinked = false
function cudahelper.link_driver_library()
    if cudaruntimelinked then return end
    local path = assert(cudapaths[ffi.os],"unknown OS?")
    terralib.linklibrary(path)
    cudaruntimelinked = true
end

--

local ef = terralib.externfunction
local externcall_builtin = terralib.externfunction

local RuntimeAPI = terralib.includec("cuda_runtime.h")
local HijackAPI = terralib.includec("legion_terra_cudart_hijack.h")

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
local DriverAPI = {
  cuInit = ef("cuInit", {uint32} -> uint32);
  cuCtxGetCurrent = ef("cuCtxGetCurrent", {&&CUctx_st} -> uint32);
  cuCtxGetDevice = ef("cuCtxGetDevice",{&int32} -> uint32);
  cuDeviceGet = ef("cuDeviceGet",{&int32,int32} -> uint32);
  cuCtxCreate_v2 = ef("cuCtxCreate_v2",{&&CUctx_st,uint32,int32} -> uint32);
  cuDeviceComputeCapability = ef("cuDeviceComputeCapability",
    {&int32,&int32,int32} -> uint32);
  cuLinkCreate_v2 = ef("cuLinkCreate_v2",
    {uint32,&uint32,&&opaque,&&CUlinkState_st} -> uint32);
  cuLinkAddData_v2 = ef("cuLinkAddData_v2",
    {&CUlinkState_st,uint32,&opaque,uint64,&int8,uint32,&uint32,&&opaque} -> uint32);
  cuLinkComplete = ef("cuLinkComplete",
    {&CUlinkState_st,&&opaque,&uint64} -> uint32);
  cuLinkDestroy = ef("cuLinkDestroy", {&CUlinkState_st} -> uint32);
}

local dlfcn = terralib.includec("dlfcn.h")
local terra has_symbol(symbol : rawstring)
  var lib = dlfcn.dlopen([&int8](0), dlfcn.RTLD_LAZY)
  var has_symbol = dlfcn.dlsym(lib, symbol) ~= [&opaque](0)
  dlfcn.dlclose(lib)
  return has_symbol
end

do
  if has_symbol("cuInit") then
    terra cudahelper.check_cuda_available()
      var r = DriverAPI.cuInit(0)
      return r == 0
    end
  else
    terra cudahelper.check_cuda_available()
      return false
    end
  end
end

-- copied and modified from cudalib.lua in Terra interpreter

local terra init_cuda() : int32
  var r = DriverAPI.cuInit(0)
  assert(r == 0, "CUDA error in cuInit")
  var cx : &CUctx_st
  r = DriverAPI.cuCtxGetCurrent(&cx)
  assert(r == 0, "CUDA error in cuCtxGetCurrent")
  var d : int32
  if cx ~= nil then
    r = DriverAPI.cuCtxGetDevice(&d)
    assert(r == 0, "CUDA error in cuCtxGetDevice")
  else
    r = DriverAPI.cuDeviceGet(&d, 0)
    assert(r == 0, "CUDA error in cuDeviceGet")
    r = DriverAPI.cuCtxCreate_v2(&cx, 0, d)
    assert(r == 0, "CUDA error in cuCtxCreate_v2")
  end

  return d
end

local terra get_cuda_version(device : int) : uint64
  var major : int, minor : int
  var r = DriverAPI.cuDeviceComputeCapability(&major, &minor, device)
  assert(r == 0, "CUDA error in cuDeviceComputeCapability")
  return [uint64](major * 10 + minor)
end

--

struct fat_bin_t {
  magic : int,
  versions : int,
  data : &opaque,
  filename : &opaque,
}

local terra register_ptx(ptxc : rawstring, ptxSize : uint32, version : uint64) : &&opaque
  var fat_bin : &fat_bin_t
  -- TODO: this line is leaking memory
  fat_bin = [&fat_bin_t](C.malloc(sizeof(fat_bin_t)))
  fat_bin.magic = 1234
  fat_bin.versions = 5678
  fat_bin.data = C.malloc(ptxSize + 1)
  fat_bin.data = ptxc
  var handle = HijackAPI.hijackCudaRegisterFatBinary(fat_bin)
  return handle
end

local terra register_function(handle : &&opaque, id : int, name : &int8)
  HijackAPI.hijackCudaRegisterFunction(handle, [&int8](id), name)
end

local function find_device_library(target)
  local device_lib_dir = terralib.cudahome .. "/nvvm/libdevice/"
  local libdevice = nil
  for f in io.popen("ls " .. device_lib_dir):lines() do
    local version = tonumber(string.match(string.match(f, "[0-9][0-9][.]"), "[0-9][0-9]"))
    if version <= target then
      libdevice = device_lib_dir .. f
    end
  end
  assert(libdevice ~= nil, "Failed to find a device library")
  return libdevice
end

function cudahelper.jit_compile_kernels_and_register(kernels)
  local module = {}
  for k, v in pairs(kernels) do
    module[v.name] = v.kernel
  end
  local device = init_cuda()
  local version = get_cuda_version(device)
  local libdevice = find_device_library(tonumber(version))
  local llvmbc = terralib.linkllvm(libdevice)
  externcall_builtin = function(name, ftype)
    return llvmbc:extern(name, ftype)
  end
  local ptx = cudalib.toptx(module, nil, version)

  local ptxc = terralib.constant(ptx)
  local handle = terralib.newsymbol(&&opaque, "handle")
  local register = quote
    var [handle] = register_ptx(ptxc, [ptx:len() + 1], [version])
  end

  for k, v in pairs(kernels) do
    register = quote
      [register]
      register_function([handle], [k], [v.name])
    end
  end

  return register
end

function cudahelper.codegen_kernel_call(kernel_id, counts, args)
  local setupArguments = terralib.newlist()

  local offset = 0
  for i = 1, #args do
    local arg =  args[i]
    local size = terralib.sizeof(arg.type)
    setupArguments:insert(quote
      RuntimeAPI.cudaSetupArgument(&[arg], size, offset)
    end)
    offset = offset + size
  end

  local grid = terralib.newsymbol(RuntimeAPI.dim3, "grid")
  local block = terralib.newsymbol(RuntimeAPI.dim3, "block")
  local launch_domain_init

  local function round_exp(v, n)
    return `((v + (n - 1)) / n)
  end

  -- TODO: Make this handle different thread block sizes and access strides
  if #counts == 1 then
    local threadSizeX = 128
    launch_domain_init = quote
      [grid].x, [grid].y, [grid].z =
        [round_exp(counts[1], threadSizeX)], 1, 1
      [block].x, [block].y, [block].z =
        threadSizeX, 1, 1
    end
  elseif #counts == 2 then
    local threadSizeX = 16
    local threadSizeY = 16
    launch_domain_init = quote
      [grid].x, [grid].y, [grid].z =
        [round_exp(counts[1], threadSizeX)],
        [round_exp(counts[2], threadSizeY)], 1
      [block].x, [block].y, [block].z =
        [threadSizeX], [threadSizeY], 1
    end
  elseif #counts == 3 then
    local threadSizeX = 16
    local threadSizeY = 8
    local threadSizeZ = 2
    launch_domain_init = quote
      [grid].x, [grid].y, [grid].z =
        [round_exp(counts[1], threadSizeX)],
        [round_exp(counts[2], threadSizeY)],
        [round_exp(counts[3], threadSizeZ)]
      [block].x, [block].y, [block].z =
        [threadSizeX], [threadSizeY], [threadSizeZ]
    end
  else
    assert(false, "Indexspaces more than 3 dimensions are not supported")
  end

  return quote
    var [grid], [block]
    [launch_domain_init]
    RuntimeAPI.cudaConfigureCall([grid], [block], 0, nil)
    [setupArguments]
    RuntimeAPI.cudaLaunch([&int8](kernel_id))
  end
end

local builtin_gpu_fns = {
  acos  = externcall_builtin("__nv_acos"  , double -> double),
  asin  = externcall_builtin("__nv_asin"  , double -> double),
  atan  = externcall_builtin("__nv_atan"  , double -> double),
  cbrt  = externcall_builtin("__nv_cbrt"  , double -> double),
  ceil  = externcall_builtin("__nv_ceil"  , double -> double),
  cos   = externcall_builtin("__nv_cos"   , double -> double),
  fabs  = externcall_builtin("__nv_fabs"  , double -> double),
  floor = externcall_builtin("__nv_floor" , double -> double),
  fmod  = externcall_builtin("__nv_fmod"  , {double, double} -> double),
  log   = externcall_builtin("__nv_log"   , double -> double),
  pow   = externcall_builtin("__nv_pow"   , {double, double} -> double),
  sin   = externcall_builtin("__nv_sin"   , double -> double),
  sqrt  = externcall_builtin("__nv_sqrt"  , double -> double),
  tan   = externcall_builtin("__nv_tan"   , double -> double),
}

local cpu_fn_to_gpu_fn = {}

function cudahelper.register_builtin(name, cpu_fn)
  cpu_fn_to_gpu_fn[cpu_fn] = builtin_gpu_fns[name]
end

function cudahelper.replace_with_builtin(fn)
  return cpu_fn_to_gpu_fn[fn] or fn
end

return cudahelper
