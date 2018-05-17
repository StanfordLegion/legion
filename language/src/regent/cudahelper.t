-- Copyright 2018 Stanford University
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

local std_base = require("regent/std_base")
local config = require("regent/config").args()
local report = require("common/report")

local cudahelper = {}
cudahelper.check_cuda_available = function() return false end

if not config["cuda"] or not terralib.cudacompile then
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
  cuCtxDestroy = ef("cuCtxDestroy",{&CUctx_st} -> uint32);
  cuDeviceComputeCapability = ef("cuDeviceComputeCapability",
    {&int32,&int32,int32} -> uint32);
}

local dlfcn = terralib.includec("dlfcn.h")
local terra has_symbol(symbol : rawstring)
  var lib = dlfcn.dlopen([&int8](0), dlfcn.RTLD_LAZY)
  var has_symbol = dlfcn.dlsym(lib, symbol) ~= [&opaque](0)
  dlfcn.dlclose(lib)
  return has_symbol
end

do
  if not config["cuda-offline"] then
    if has_symbol("cuInit") then
      local r = DriverAPI.cuInit(0)
      assert(r == 0)
      terra cudahelper.check_cuda_available()
        return [r] == 0;
      end
    else
      terra cudahelper.check_cuda_available()
        return false
      end
    end
  else
    terra cudahelper.check_cuda_available()
      return true
    end
  end
end

-- copied and modified from cudalib.lua in Terra interpreter

local c = terralib.includec("unistd.h")

local lua_assert = assert
local terra assert(x : bool, message : rawstring)
  if not x then
    var stderr = C.fdopen(2, "w")
    C.fprintf(stderr, "assertion failed: %s\n", message)
    -- Just because it's stderr doesn't mean it's unbuffered...
    C.fflush(stderr)
    C.abort()
  end
end

local terra get_cuda_version() : uint64
  var cx : &CUctx_st
  var cx_created = false
  var r = DriverAPI.cuCtxGetCurrent(&cx)
  assert(r == 0, "CUDA error in cuCtxGetCurrent")
  var device : int32
  if cx ~= nil then
    r = DriverAPI.cuCtxGetDevice(&device)
    assert(r == 0, "CUDA error in cuCtxGetDevice")
  else
    r = DriverAPI.cuDeviceGet(&device, 0)
    assert(r == 0, "CUDA error in cuDeviceGet")
    r = DriverAPI.cuCtxCreate_v2(&cx, 0, device)
    assert(r == 0, "CUDA error in cuCtxCreate_v2")
    cx_created = true
  end

  var major : int, minor : int
  r = DriverAPI.cuDeviceComputeCapability(&major, &minor, device)
  assert(r == 0, "CUDA error in cuDeviceComputeCapability")
  var version = [uint64](major * 10 + minor)
  if cx_created then
    DriverAPI.cuCtxDestroy(cx)
  end
  return version
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

local supported_archs = {
  ["fermi"]   = 20,
  ["kepler"]  = 30,
  ["k20"]     = 35,
  ["maxwell"] = 52,
  ["pascal"]  = 60,
  ["volta"]   = 70,
}

local function parse_cuda_arch(arch)
  arch = string.lower(arch)
  local sm = supported_archs[arch]
  if sm == nil then
    local archs
    for k, v in pairs(supported_archs) do
      archs = (not archs and k) or (archs and archs .. ", " .. k)
    end
    print("Error: Unsupported GPU architecture " .. arch ..
          ". Supported architectures: " .. archs)
    os.exit(1)
  end
  return sm
end

function cudahelper.jit_compile_kernels_and_register(kernels)
  local module = {}
  for k, v in pairs(kernels) do
    module[v.name] = v.kernel
  end
  local version
  if not config["cuda-offline"] then
    version = get_cuda_version()
  else
    version = parse_cuda_arch(config["cuda-arch"])
  end
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

local THREAD_BLOCK_SIZE = 128
local MAX_NUM_BLOCK = 32768
local GLOBAL_RED_BUFFER = 256

local tid_x   = cudalib.nvvm_read_ptx_sreg_tid_x
local n_tid_x = cudalib.nvvm_read_ptx_sreg_ntid_x
local bid_x   = cudalib.nvvm_read_ptx_sreg_ctaid_x
local n_bid_x = cudalib.nvvm_read_ptx_sreg_nctaid_x

local tid_y   = cudalib.nvvm_read_ptx_sreg_tid_y
local n_tid_y = cudalib.nvvm_read_ptx_sreg_ntid_y
local bid_y   = cudalib.nvvm_read_ptx_sreg_ctaid_y
local n_bid_y = cudalib.nvvm_read_ptx_sreg_nctaid_y

local tid_z   = cudalib.nvvm_read_ptx_sreg_tid_z
local n_tid_z = cudalib.nvvm_read_ptx_sreg_ntid_z
local bid_z   = cudalib.nvvm_read_ptx_sreg_ctaid_z
local n_bid_z = cudalib.nvvm_read_ptx_sreg_nctaid_z

local barrier = cudalib.nvvm_barrier0

local supported_scalar_red_ops = {
  ["+"]   = true,
  ["*"]   = true,
  ["max"] = true,
  ["min"] = true,
}

function cudahelper.global_thread_id()
  --local bid = `(bid_x() + n_bid_x() * bid_y() + n_bid_x() * n_bid_y() * bid_z())
  --local num_threads = `(n_tid_x() * n_tid_y() * n_tid_z())
  --return `([bid] * [num_threads] +
  --         tid_x() +
  --         n_tid_x() * tid_y() +
  --         n_tid_x() * n_tid_y() * tid_z())
  local bid = `(bid_x() + n_bid_x() * bid_y() + n_bid_x() * n_bid_y() * bid_z())
  local num_threads = `(n_tid_x())
  return `([bid] * [num_threads] + tid_x())
end

function cudahelper.global_block_id()
  return `(bid_x() + n_bid_x() * bid_y() + n_bid_x() * n_bid_y() * bid_z())
end

-- Slow atomic operation implementations (copied and modified from Ebb)
local terra cas_uint64(address : &uint64, compare : uint64, value : uint64)
  return terralib.asm(terralib.types.uint64,
                      "atom.global.cas.b64 $0, [$1], $2, $3;",
                      "=l,l,l,l", true, address, compare, value)
end
cas_uint64:setinlined(true)

local terra cas_uint32(address : &uint32, compare : uint32, value : uint32)
  return terralib.asm(terralib.types.uint32,
                      "atom.global.cas.b32 $0, [$1], $2, $3;",
                      "=r,l,r,r", true, address, compare, value)
end
cas_uint32:setinlined(true)

local function generate_atomic(op, typ)
  if op == "+" and typ == float then
    return terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32",
                              {&float,float} -> {float})
  end

  local cas_type
  local cas_func
  if sizeof(typ) == 4 then
    cas_type = uint32
    cas_func = cas_uint32
  else
    lua_assert(sizeof(typ) == 8)
    cas_type = uint64
    cas_func = cas_uint64
  end
  local terra atomic_op(address : &typ, operand : typ)
    var old : typ = @address
    var assumed : typ
    var new     : typ

    var new_b     : &cas_type = [&cas_type](&new)
    var assumed_b : &cas_type = [&cas_type](&assumed)
    var res       :  cas_type

    var mask = false
    repeat
      if not mask then
        assumed = old
        new     = [std_base.quote_binary_op(op, assumed, operand)]
        res     = cas_func([&cas_type](address), @assumed_b, @new_b)
        old     = @[&typ](&res)
        mask    = assumed == old
      end
    until mask
  end
  atomic_op:setinlined(true)
  return atomic_op
end

local vprintf = ef("cudart:vprintf", {&int8,&int8} -> int)

local function createbuffer(args)
  local Buf = terralib.types.newstruct()
  for i,e in ipairs(args) do
    local typ = e:gettype()
    local field = "_"..tonumber(i)
    typ = typ == float and double or typ
    table.insert(Buf.entries,{field,typ})
  end
  return quote
    var buf : Buf
    escape
        for i,e in ipairs(args) do
            emit quote
               buf.["_"..tonumber(i)] = e
            end
        end
    end
  in
    [&int8](&buf)
  end
end

local cuda_printf = macro(function(fmt,...)
  local buf = createbuffer({...})
  return `vprintf(fmt,buf)
end)

function cudahelper.compute_reduction_buffer_size(node, reductions)
  local size = 0
  for k, v in pairs(reductions) do
    if size ~= 0 then
      -- TODO: We assume there is only one scalar reduction for now
      report.error(node,
          "Multiple scalar reductions in a CUDA task are not supported yet")
    elseif not supported_scalar_red_ops[v] then
      report.error(node,
          "Scalar reduction with operator " .. v .. " is not supported yet")
    elseif not (sizeof(k.type) == 4 or sizeof(k.type) == 8) then
      report.error(node,
          "Scalar reduction for type " .. tostring(k.type) .. " is not supported yet")
    end
    size = size + THREAD_BLOCK_SIZE * sizeof(k.type)
  end
  return size
end

function cudahelper.generate_reduction_preamble(reductions)
  local preamble = quote end
  local device_ptrs = terralib.newlist()
  local device_ptrs_map = {}

  for red_var, red_op in pairs(reductions) do
    local device_ptr = terralib.newsymbol(&red_var.type, red_var.displayname)
    local init = std_base.reduction_op_init[red_op][red_var.type]
    preamble = quote
      [preamble];
      var [device_ptr] = [&red_var.type](nil)
      do
        var r = RuntimeAPI.cudaMalloc([&&opaque](&[device_ptr]),
                                      [sizeof(red_var.type) * GLOBAL_RED_BUFFER])
        assert([r] == 0 and [device_ptr] ~= [&red_var.type](nil), "cudaMalloc failed")
        var v : (red_var.type)[GLOBAL_RED_BUFFER]
        for i = 0, GLOBAL_RED_BUFFER do v[i] = [init] end
        RuntimeAPI.cudaMemcpy([device_ptr], [&opaque]([&red_var.type](v)),
                              [sizeof(red_var.type) * GLOBAL_RED_BUFFER],
                              RuntimeAPI.cudaMemcpyHostToDevice)
      end
    end
    device_ptrs:insert(device_ptr)
    device_ptrs_map[device_ptr] = red_var
  end

  return device_ptrs, device_ptrs_map, preamble
end

function cudahelper.generate_reduction_kernel(reductions, device_ptrs_map)
  local preamble = quote end
  local postamble = quote end
  for device_ptr, red_var in pairs(device_ptrs_map) do
    local red_op = reductions[red_var]
    local shared_mem_ptr =
      cudalib.sharedmemory(red_var.type, THREAD_BLOCK_SIZE)
    local init = std_base.reduction_op_init[red_op][red_var.type]
    preamble = quote
      [preamble]
      var [red_var] = [init]
      [shared_mem_ptr][ tid_x() ] = [red_var]
    end

    local tid = terralib.newsymbol(c.size_t, "tid")
    local reduction_tree = quote end
    local step = THREAD_BLOCK_SIZE
    while step > 64 do
      step = step / 2
      reduction_tree = quote
        [reduction_tree]
        if [tid] < step then
          var v = [std_base.quote_binary_op(red_op,
                                            `([shared_mem_ptr][ [tid] ]),
                                            `([shared_mem_ptr][ [tid] + [step] ]))]

          terralib.attrstore(&[shared_mem_ptr][ [tid] ], v, { isvolatile = true })
        end
        barrier()
      end
    end
    local unrolled_reductions = terralib.newlist()
    while step > 1 do
      step = step / 2
      unrolled_reductions:insert(quote
        do
          var v = [std_base.quote_binary_op(red_op,
                                            `([shared_mem_ptr][ [tid] ]),
                                            `([shared_mem_ptr][ [tid] + [step] ]))]
          terralib.attrstore(&[shared_mem_ptr][ [tid] ], v, { isvolatile = true })
        end
        barrier()
      end)
    end
    reduction_tree = quote
      [reduction_tree]
      if [tid] < 32 then
        [unrolled_reductions]
      end
    end
    postamble = quote
      do
        var [tid] = tid_x()
        var bid = [cudahelper.global_block_id()]
        [shared_mem_ptr][ [tid] ] = [red_var]
        barrier()
        [reduction_tree]
        if [tid] == 0 then
          [generate_atomic(red_op, red_var.type)](
            &[device_ptr][bid % [GLOBAL_RED_BUFFER] ], [shared_mem_ptr][ [tid] ])
        end
      end
    end
  end
  return preamble, postamble
end

function cudahelper.generate_reduction_postamble(reductions, device_ptrs_map)
  local postamble = quote end
  for device_ptr, red_var in pairs(device_ptrs_map) do
    local red_op = reductions[red_var]
    local init = std_base.reduction_op_init[red_op][red_var.type]
    postamble = quote
      [postamble]
      do
        var v : (red_var.type)[GLOBAL_RED_BUFFER]
        RuntimeAPI.cudaMemcpy([&opaque]([&red_var.type](v)), [device_ptr],
                              [sizeof(red_var.type) * GLOBAL_RED_BUFFER],
                              RuntimeAPI.cudaMemcpyDeviceToHost)
        var tmp : red_var.type = [init]
        for i = 0, GLOBAL_RED_BUFFER do
          tmp = [std_base.quote_binary_op(red_op, tmp, `(v[i]))]
        end
        [red_var] = [std_base.quote_binary_op(red_op, red_var, tmp)]
        RuntimeAPI.cudaFree([device_ptr])
      end
    end
  end
  return postamble
end

function cudahelper.codegen_kernel_call(kernel_id, count, args, shared_mem_size)
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

  local function round_exp(v, n)
    return `((v + (n - 1)) / n)
  end

  local launch_domain_init = quote
    [block].x, [block].y, [block].z = THREAD_BLOCK_SIZE, 1, 1
    var num_blocks = [round_exp(count, THREAD_BLOCK_SIZE)]
    if num_blocks <= MAX_NUM_BLOCK then
      [grid].x, [grid].y, [grid].z = num_blocks, 1, 1
    elseif [count] / MAX_NUM_BLOCK <= MAX_NUM_BLOCK then
      [grid].x, [grid].y, [grid].z =
        MAX_NUM_BLOCK, [round_exp(num_blocks, MAX_NUM_BLOCK)], 1
    else
      [grid].x, [grid].y, [grid].z =
        MAX_NUM_BLOCK, MAX_NUM_BLOCK,
        [round_exp(num_blocks, MAX_NUM_BLOCK, MAX_NUM_BLOCK)]
    end
  end

  return quote
    var [grid], [block]
    [launch_domain_init]
    RuntimeAPI.cudaConfigureCall([grid], [block], shared_mem_size, nil)
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
