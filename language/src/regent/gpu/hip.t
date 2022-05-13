-- Copyright 2022 Stanford University
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

-- HIP-specific Settings for the Regent GPU Code Generator
--
-- IMPORTANT: DO NOT import this file directly, instead see
-- gpu/helper.t for usage.

local base = require("regent/std_base")
local common = require("regent/gpu/common")

local hiphelper = {}

local RuntimeAPI = terralib.includecstring [[
#define __HIP_PLATFORM_HCC__ 1
#include <hip/hip_runtime.h>
hipStream_t hipGetTaskStream();
//#include "realm/hip/hiphijack_api.h"
]]

function hiphelper.check_gpu_available()
  -- FIXME: actually check HIP availability.
  return true
end

-- Exit early if it's not available.
if not hiphelper.check_gpu_available() then
  return hiphelper
end

do
  local linked = false
  function hiphelper.link_driver_library()
    if linked then return end
    base.linklibrary("libamdhip64.so")
    linked = true
  end
end

function hiphelper.driver_library_link_flags()
  -- If the hijack is turned off, we need extra dependencies to link
  -- the generated HIP code correctly
  if base.c.REGENT_USE_HIJACK == 0 then
    local hip_path = os.getenv("HIP_PATH")
    return terralib.newlist({"-L" .. hip_path .. "/lib", "-lamdhip64"})
  end
  return terralib.newlist()
end

local ensure_arch
do
  local arch
  function ensure_arch()
    if arch then return arch end

    arch = base.config["gpu-arch"]
    if arch == "unspecified" then
      print("need to specify GPU arch via -fgpu-arch")
      assert(false)
    end
    return arch
  end
end

local ensure_target
do
  local target
  function ensure_target()
    if target then return target end

    local arch = ensure_arch()
    target = terralib.newtarget {
      Triple = 'amdgcn-amd-amdhsa',
      CPU = arch,
      FloatABIHard = true,
    }
    return target
  end
end

local ensure_device_paths
local ensure_device_extern
do
  local paths
  local libs
  local extern

  function ensure_device_paths()
    if paths then return paths end

    local paths = {}

    local rocm_path = os.getenv('ROCM_PATH')
    assert(rocm_path, "Please set ROCM_PATH to root of ROCm installation")
    local rocm_bincode_path = rocm_path .. "/amdgcn/bitcode"

    local arch = ensure_arch()
    assert(string.sub(arch, 1, 3) == "gfx")
    local arch_id = string.sub(arch, 4)

    -- https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/1de32ef43a44cf578ea5fa351df6b6da12ba84c4/doc/OCML.md#controls
    local options = {
      correctly_rounded_sqrt = true, -- float sqrt must be correctly rounded
      daz_opt = false,               -- subnormal values may not be flushed to zero
      finite_only = false,           -- inf and nan values may be produced
      unsafe_math = false,           -- maintain higher accuracy results
      wavefrontsize64 = true,        -- all current devices use a wavefront size of 64
    }

    for k, v in pairs(options) do
      paths[k] = rocm_bincode_path .. "/oclc_" .. k .. "_" .. (v and "on" or "off") .. ".bc"
    end

    paths["isa_version"] = rocm_bincode_path .. "/oclc_isa_version_" .. arch_id .. ".bc"

    paths["ocml"] = rocm_bincode_path .. "/ocml.bc"

    return paths
  end

  local function ensure_device_libs()
    if libs then return libs end

    local libs = {}

    local target = ensure_target()

    for k, v in pairs(ensure_device_paths()) do
      libs[k] = terralib.linkllvm(v, target)
    end

    return libs
  end

  function ensure_device_extern()
    if extern then return extern end

    local libs = ensure_device_libs()

    local ocml = libs["ocml"]

    function extern(name, type)
      return ocml:extern(name, type)
    end

    return extern
  end
end

local function pr(...)
  print(...)
  return ...
end

local terra check(ok : RuntimeAPI.hipError_t, location : rawstring)
  if ok ~= RuntimeAPI.HIP_SUCCESS then
    base.c.printf("error in %s: %s\n", location, RuntimeAPI.hipGetErrorName(ok))
    base.c.abort()
  end
end

function hiphelper.jit_compile_kernels_and_register(kernels)
  for _, kernel in ipairs(kernels) do
    kernel.kernel:setcallingconv("amdgpu_kernel")
  end

  local module = {}
  kernels:map(function(kernel) module[kernel.name] = kernel.kernel end)

  local arch = ensure_arch()
  local amd_target = ensure_target()

  if base.config["hip-pretty-kernels"] then
    for _, kernel in ipairs(kernels) do
      print()
      print('################################################################')
      print('################################################################')
      print('################################################################')
      print()

      kernel.kernel:printpretty(false)

      print()

      print(terralib.saveobj(nil, "llvmir", {[kernel.name]=kernel.kernel}, {}, amd_target, false))
      local obj = os.tmpname()
      terralib.saveobj(obj, "object", {[kernel.name]=kernel.kernel}, {}, amd_target)
      os.remove(obj)
    end
  end

  local device_o = os.tmpname()
  local device_so = os.tmpname()
  local bundle_o = os.tmpname()

  terralib.saveobj(device_o, "object", module, {}, amd_target)

  local device_paths = ""
  for k, v in pairs(ensure_device_paths()) do
    device_paths = device_paths .. " " .. v
  end

  local hip_path = os.getenv("HIP_PATH")
  local rocm_path = os.getenv("ROCM_PATH") or (hip_path and hip_path .. "/..")
  local llvm_path = rocm_path and (rocm_path .. "/llvm/bin/") or ""

  os.execute(pr(llvm_path .. "ld.lld -shared -plugin-opt=mcpu=" .. arch .. " -plugin-opt=-amdgpu-internalize-symbols -plugin-opt=O3 -plugin-opt=-amdgpu-early-inline-all=true -plugin-opt=-amdgpu-function-calls=false -o " .. device_so .. " " .. device_o .. " " .. device_paths))

  os.execute(pr(llvm_path .. "clang-offload-bundler --inputs=/dev/null," .. device_so .. " --type=o --outputs=" .. bundle_o .. " --targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--" .. arch))

  local f = assert(io.open(bundle_o, "rb"))
  local bundle_contents = f:read("*all")
  f:close()

  os.remove(device_o)
  os.remove(device_so)
  os.remove(bundle_o)

  local register = quote
    var module : RuntimeAPI.hipModule_t
    check(RuntimeAPI.hipModuleLoadData(&module, bundle_contents), "hipModuleLoadData")
    escape
      for _, k in ipairs(kernels) do
        local kernel = k.kernel

        local func = kernel.hip_func
        assert(func) -- Hopefully this is always true. If not, then it means we're generating kernels that we don't call anywhere.

        emit quote
          check(RuntimeAPI.hipModuleGetFunction(&func, module, k.name), "hipModuleGetFunction")
        end
      end
    end
  end

  return register
end

-- #####################################
-- ## Primitives
-- #################

local THREAD_BLOCK_SIZE = 128
local NUM_THREAD_X = 16
local NUM_THREAD_Y = THREAD_BLOCK_SIZE / NUM_THREAD_X
local MAX_NUM_BLOCK = 32768
local MAX_SIZE_INLINE_KERNEL_PARAMS = 1024
local GLOBAL_RED_BUFFER = 256
assert(GLOBAL_RED_BUFFER % THREAD_BLOCK_SIZE == 0)

hiphelper.THREAD_BLOCK_SIZE = THREAD_BLOCK_SIZE
hiphelper.NUM_THREAD_X = NUM_THREAD_X
hiphelper.NUM_THREAD_Y = NUM_THREAD_Y
hiphelper.MAX_SIZE_INLINE_KERNEL_PARAMS = MAX_SIZE_INLINE_KERNEL_PARAMS
hiphelper.GLOBAL_RED_BUFFER = GLOBAL_RED_BUFFER

local tid_x = terralib.intrinsic("llvm.amdgcn.workitem.id.x",{} -> int32)
local bid_x = terralib.intrinsic("llvm.amdgcn.workgroup.id.x",{} -> int32)
local tid_y = terralib.intrinsic("llvm.amdgcn.workitem.id.y",{} -> int32)
local bid_y = terralib.intrinsic("llvm.amdgcn.workgroup.id.y",{} -> int32)
local tid_z = terralib.intrinsic("llvm.amdgcn.workitem.id.z",{} -> int32)
local bid_z = terralib.intrinsic("llvm.amdgcn.workgroup.id.z",{} -> int32)

hiphelper.tid_x = tid_x
hiphelper.tid_y = tid_y
hiphelper.tid_z = tid_z
hiphelper.bid_x = bid_x
hiphelper.bid_y = bid_y
hiphelper.bid_z = bid_z

local hip_addrspace = {
  generic = 0,
  global = 1,
  region = 2,
  ["local"] = 3,
  constant = 4,
  private = 5,
}

function constant_ptr(typ)
  return terralib.types.pointer(typ, hip_addrspace.constant)
end

local dispatch_ptr = terralib.intrinsic("llvm.amdgcn.dispatch.ptr", {}->constant_ptr(int8))

function make_n_tid(dim, idx)
  local terra n_tid(dp : constant_ptr(int8))
    return ([constant_ptr(int16)](dp))[idx]
  end
  n_tid:setname("n_tid_" .. dim)
  n_tid:setinlined(true)
  return n_tid
end

local n_tid_x = make_n_tid("x", 2)
local n_tid_y = make_n_tid("y", 3)
local n_tid_z = make_n_tid("z", 4)

function make_n_bid(dim, idx)
  local terra n_bid(dp : constant_ptr(int8))
    return ([constant_ptr(int32)](dp))[idx]
  end
  n_bid:setname("n_bid_" .. dim)
  n_bid:setinlined(true)
  return n_bid
end

local n_bid_x = make_n_bid("x", 3)
local n_bid_y = make_n_bid("y", 4)
local n_bid_z = make_n_bid("z", 5)

local raw_barrier = terralib.intrinsic("llvm.amdgcn.s.barrier", {}->{})
terra barrier()
  -- FIXME: probably synchronizing more aggressively than required, but the syncscope seems to be broken
  terralib.fence({ordering="release"})--, syncscope="workgroup"})
  raw_barrier()
  terralib.fence({ordering="acquire"})--, syncscope="workgroup"})
end
barrier:setinlined(true)

hiphelper.barrier = barrier

function hiphelper.global_thread_id()
  return quote
    var dp = dispatch_ptr()
    -- Important: HIP grid sizes come **pre-multiplied** so we have to
    -- divide out the block size to get the number of blocks.
    var nx : int64 = n_bid_x(dp)/n_tid_x(dp)
    var ny : int64 = n_bid_y(dp)/n_tid_y(dp)
    var bid : int64 = bid_x() + nx * bid_y() + nx * ny * bid_z()
    var num_threads : int64 = n_tid_x(dp)
  in
    bid * num_threads + tid_x()
  end
end

function hiphelper.global_thread_id_flat()
  return quote
    var dp = dispatch_ptr()
  in
    tid_x() + bid_x() * int64(n_tid_x(dp))
  end
end

function hiphelper.global_block_id()
  return quote
    var dp = dispatch_ptr()
    var nx : int64 = n_bid_x(dp)/n_tid_x(dp)
    var ny : int64 = n_bid_y(dp)/n_tid_y(dp)
  in
    bid_x() + nx * bid_y() + nx * ny * bid_z()
  end
end

function hiphelper.sharedmemory(typ, size)
  return terralib.global(typ[size], nil, nil, size == 0, false, hip_addrspace["local"])
end

function hiphelper.generate_atomic_update(op, typ)
  local atomic_op = common.generate_atomic_update(op, typ)
  assert(atomic_op)
  return atomic_op
end

-- #####################################
-- ## Code generation for kernel launch
-- #################

local get_hip_fn_name
do
  local suffixes = {
    [float] = "f32",
    [double] = "f64",
  }
  function get_hip_fn_name(name, type)
    -- https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/1de32ef43a44cf578ea5fa351df6b6da12ba84c4/doc/OCML.md#naming-convention
    local suffix = suffixes[type]
    if not suffix then
      print("don't know how to generate function for " .. tostring(type))
      assert(false)
    end
    return "__ocml_" .. name .. "_" .. suffix
  end
end

local function get_hip_definition(self)
  if self:has_variant("hip") then
    return self:get_variant("hip")
  else
    local fn_type = self.super:get_definition().type
    local fn_name = get_hip_fn_name(self:get_name(), self:get_arg_type())
    assert(fn_name ~= nil)
    local externcall_builtin = ensure_device_extern()
    local fn = externcall_builtin(fn_name, fn_type)
    self:set_variant("hip", fn)
    return fn
  end
end

function hiphelper.get_gpu_variant(math_fn)
  return math_fn:override(get_hip_definition)
end

function hiphelper.codegen_kernel_call(cx, kernel, count, args, shared_mem_size, tight)
  local setupArguments = terralib.newlist()

  local arglen = common.count_arguments(args)
  local arg_arr = terralib.newsymbol((&opaque)[arglen], "__args")
  setupArguments:insert(quote var [arg_arr]; end)
  local idx = 0
  for i = 1, #args do
    local arg = args[i]
    -- Need to flatten the arguments into individual primitive values
    idx = common.generate_arg_setup(setupArguments, arg_arr, arg, arg.type, idx)
  end

  local grid = terralib.newsymbol(RuntimeAPI.dim3, "grid")
  local block = terralib.newsymbol(RuntimeAPI.dim3, "block")
  local num_blocks = terralib.newsymbol(int64, "num_blocks")

  local stream = terralib.newsymbol(RuntimeAPI.hipStream_t, "stream")

  if not kernel.hip_func then
    kernel.hip_func = terralib.global(RuntimeAPI.hipFunction_t, nil, kernel.name .. "_func")
  end
  local func = kernel.hip_func

  local function round_exp(v, n)
    return `((v + (n - 1)) / n)
  end

  local launch_domain_init = nil
  if not cx.use_2d_launch then
    launch_domain_init = quote
      if [count] <= THREAD_BLOCK_SIZE and tight then
        [block].x, [block].y, [block].z = [count], 1, 1
      else
        [block].x, [block].y, [block].z = THREAD_BLOCK_SIZE, 1, 1
      end
      var [num_blocks] = [round_exp(count, THREAD_BLOCK_SIZE)]
    end
  else
    launch_domain_init = quote
      if [count] <= NUM_THREAD_X and tight then
        [block].x, [block].y, [block].z = [count], NUM_THREAD_Y, 1
      else
        [block].x, [block].y, [block].z = NUM_THREAD_X, NUM_THREAD_Y, 1
      end
      var [num_blocks] = [round_exp(count, NUM_THREAD_X)]
    end
  end

  launch_domain_init = quote
    [launch_domain_init]
    if [num_blocks] <= MAX_NUM_BLOCK then
      [grid].x, [grid].y, [grid].z = [num_blocks], 1, 1
    elseif [num_blocks] / MAX_NUM_BLOCK <= MAX_NUM_BLOCK then
      [grid].x, [grid].y, [grid].z =
        MAX_NUM_BLOCK, [round_exp(num_blocks, MAX_NUM_BLOCK)], 1
    else
      [grid].x, [grid].y, [grid].z =
        MAX_NUM_BLOCK, MAX_NUM_BLOCK,
        [round_exp(num_blocks, MAX_NUM_BLOCK * MAX_NUM_BLOCK)]
    end
  end

  return quote
    if [count] > 0 then
      var [grid], [block]
      var [stream] = RuntimeAPI.hipGetTaskStream()
      [launch_domain_init]
      [setupArguments]
      [check](
        [RuntimeAPI.hipModuleLaunchKernel](
          [func],
          [grid].x, [grid].y, [grid].z,
          [block].x, [block].y, [block].z,
          -- Static shared memory is calculated separately in HIP,
          -- don't need dynamic shared memory.
          0 --[[ [shared_mem_size] ]], [stream], [arg_arr], nil),
        "hipModuleLaunchKernel")
    end
  end
end

terra hiphelper.device_synchronize()
  RuntimeAPI.hipDeviceSynchronize()
end

return hiphelper
