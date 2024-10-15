-- Copyright 2024 Stanford University, Los Alamos National Laboratory
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

-- CUDA-specific Settings for the Regent GPU Code Generator
--
-- IMPORTANT: DO NOT import this file directly, instead see
-- gpu/helper.t for usage.

local ast = require("regent/ast")
local base = require("regent/std_base")
local common = require("regent/gpu/common")
local data = require("common/data")
local profile = require("regent/profile")
local report = require("common/report")

local cudahelper = {}

local c = base.c
local ef = terralib.externfunction
local externcall_builtin = terralib.externfunction
local cudapaths = { OSX = "/usr/local/cuda/lib/libcuda.dylib";
                    Linux =  "libcuda.so";
                    Windows = "nvcuda.dll"; }

-- #####################################
-- ## CUDA Device API
-- #################

local struct CUctx_st
local struct CUfunc_st
local struct CUlinkState_st
local struct CUmod_st

local CUcontext = &CUctx_st
local CUfunction = &CUfunc_st
local CUjit_option = uint32
local CUlinkState = &CUlinkState_st
local CUmodule = &CUmod_st
local CUdevice = int32
local CUjit_option = uint32
local CUresult = uint32

local CU_JIT_ERROR_LOG_BUFFER = 5
local CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
local CU_JIT_INPUT_PTX = 1
local CU_JIT_TARGET = 9
local DriverAPI = {
  CU_JIT_ERROR_LOG_BUFFER = 5;
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
  CU_JIT_INPUT_PTX = 1;
  CU_JIT_TARGET = 9;

  CUcontext = CUcontext;
  CUfunction = CUfunction;
  CUjit_option = CUjit_option;
  CUlinkState = CUlinkState;
  CUmodule = CUmodule;
  CUresult = CUresult;

  cuInit = ef("cuInit", {uint32} -> CUresult);
  cuCtxGetCurrent = ef("cuCtxGetCurrent", {&CUcontext} -> CUresult);
  cuCtxGetDevice = ef("cuCtxGetDevice",{&CUdevice} -> CUresult);
  cuDeviceGet = ef("cuDeviceGet",{&CUdevice,int32} -> CUresult);
  cuCtxCreate_v2 = ef("cuCtxCreate_v2",{&CUcontext,uint32,int32} -> CUresult);
  cuCtxDestroy = ef("cuCtxDestroy",{CUcontext} -> CUresult);
  cuDeviceComputeCapability = ef("cuDeviceComputeCapability",
    {&int32,&int32,int32} -> CUresult);
  cuGetErrorName = ef("cuGetErrorName", {CUresult, &&opaque} -> CUresult);
  cuGetErrorString = ef("cuGetErrorString", {CUresult, &&opaque} -> CUresult);
  cuLinkCreate_v2 = ef("cuLinkCreate_v2",{uint32,&uint32,&&opaque,&CUlinkState} -> CUresult);
  cuLinkAddData_v2 = ef("cuLinkAddData_v2",
    {CUlinkState,uint32,&opaque,uint64,&int8,uint32,&uint32,&&opaque} -> CUresult);
  cuLinkComplete = ef("cuLinkComplete",{CUlinkState,&&opaque,&uint64} -> CUresult);
  cuLinkDestroy = ef("cuLinkDestroy",{CUlinkState} -> CUresult);
  cuModuleLoadData = ef("cuModuleLoadData",{&CUmodule,&opaque} -> CUresult);
}

local RuntimeAPI = false
do
  local function detect_cuda()
    if not terralib.cudacompile then
      return false, "Terra is built without CUDA support"
    end

    if base.c.REGENT_USE_CUDA ~= 1 then
      return false, "Legion is built without CUDA support"
    end

    -- Try to load the CUDA runtime header
    local ok = pcall(function() RuntimeAPI = terralib.includec("cuda_runtime.h") end)
    if not ok then
      return false, "cuda_runtime.h does not exist in INCLUDE_PATH"
    end

    if base.config["offline"] or base.config["gpu-offline"] then
      return true
    end

    -- Check if CUDA can actually be initialized
    local dlfcn = terralib.includec("dlfcn.h")
    local terra has_symbol(symbol : rawstring)
      var lib = dlfcn.dlopen([&int8](0), dlfcn.RTLD_LAZY)
      var has_symbol = dlfcn.dlsym(lib, symbol) ~= [&opaque](0)
      dlfcn.dlclose(lib)
      return has_symbol
    end

    if not has_symbol("cuInit") then
      return false, "the cuInit function is missing (may indicate a broken build)"
    end

    if DriverAPI.cuInit(0) ~= 0 then
      return false, "calling cuInit(0) failed for some reason (CUDA devices might not exist)"
    end

    return true
  end
  local enabled, message = detect_cuda()
  function cudahelper.check_gpu_available()
    return enabled, message
  end
end

-- Exit early if it's not available.
if not cudahelper.check_gpu_available() then
  return cudahelper
end

do
  local ffi = require('ffi')
  local cudaruntimelinked = false
  function cudahelper.link_driver_library()
    if cudaruntimelinked then return end
    local path = assert(cudapaths[ffi.os],"unknown OS?")
    base.linklibrary(path)
    cudaruntimelinked = true
  end
end

function cudahelper.driver_library_link_flags()
  -- If the hijack is turned off, we need extra dependencies to link
  -- the generated CUDA code correctly
  if base.c.REGENT_USE_HIJACK == 0 then
    return terralib.newlist({
      "-L" .. terralib.cudahome .. "/lib64", "-lcudart",
      "-L" .. terralib.cudahome .. "/lib64/stubs", "-lcuda",
      "-lpthread", "-lrt"
    })
  end
  return terralib.newlist()
end

-- #####################################
-- ## Printf for CUDA (not exposed to the user for the moment)
-- #################

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

-- #####################################
-- ## Supported CUDA compute versions
-- #################

local supported_archs = {
  ["fermi"]   = 20,
  ["kepler"]  = 32,
  ["k20"]     = 35,
  ["maxwell"] = 52,
  ["pascal"]  = 60,
  ["volta"]   = 70,
  ["turing"]  = 75,
  ["ampere"]  = 80,
}

local function parse_cuda_arch(arch)
  -- If the user manually passed a compute version, just return that
  local arch_value = tonumber(arch)
  if arch_value ~= nil then
    return arch_value
  end

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

-- #####################################
-- ## Registration functions
-- #################

local get_cuda_version
do
  local cached_cuda_version = nil
  local terra get_cuda_version_terra() : uint64
    var cx : DriverAPI.CUcontext
    var cx_created = false
    var r = DriverAPI.cuCtxGetCurrent(&cx)
    base.assert(r == 0, "CUDA error in cuCtxGetCurrent")
    var device : int32
    if cx ~= nil then
      r = DriverAPI.cuCtxGetDevice(&device)
      base.assert(r == 0, "CUDA error in cuCtxGetDevice")
    else
      r = DriverAPI.cuDeviceGet(&device, 0)
      base.assert(r == 0, "CUDA error in cuDeviceGet")
      r = DriverAPI.cuCtxCreate_v2(&cx, 0, device)
      base.assert(r == 0, "CUDA error in cuCtxCreate_v2")
      cx_created = true
    end

    var major : int, minor : int
    r = DriverAPI.cuDeviceComputeCapability(&major, &minor, device)
    base.assert(r == 0, "CUDA error in cuDeviceComputeCapability")
    var version = [uint64](major * 10 + minor)
    if cx_created then
      DriverAPI.cuCtxDestroy(cx)
    end
    return version
  end

  get_cuda_version = function()
    if cached_cuda_version ~= nil then
      return cached_cuda_version
    end
    if not (base.config["offline"] or base.config["gpu-offline"]) then
      cached_cuda_version = get_cuda_version_terra()
    else
      cached_cuda_version = parse_cuda_arch(base.config["gpu-arch"])
    end
    return cached_cuda_version
  end
end

local terra check(ok : DriverAPI.CUresult, location : rawstring)
  if ok ~= DriverAPI.CUDA_SUCCESS then
    var error_name : &opaque = nil
    var error_string : &opaque = nil
    DriverAPI.cuGetErrorName(ok, &error_name)
    DriverAPI.cuGetErrorString(ok, &error_string)
    base.c.printf("error in %s (%s): %s\n", location, error_name, error_string)
    base.c.abort()
  end
end

local terra checkrt(ok : RuntimeAPI.cuda_Error_t, location : rawstring)
  if ok ~= RuntimeAPI.cudaSuccess then
    base.c.printf("error in %s (%s): %s\n", location, RuntimeAPI.cudaGetErrorName(ok), RuntimeAPI.cudaGetErrorString(ok))
    base.c.abort()
  end
end

local struct cubin_t {
  data : rawstring,
  size : uint64
}

local terra ptx_to_cubin(ptx : rawstring, ptx_sz : uint64, version : uint64)
  var linkState : DriverAPI.CUlinkState
  var cubin : &opaque
  var cubinSize : uint64
  var error_str : rawstring = nil
  var error_sz : uint64 = 0

  var options = arrayof(
    DriverAPI.CUjit_option,
    DriverAPI.CU_JIT_TARGET,
    DriverAPI.CU_JIT_ERROR_LOG_BUFFER,
    DriverAPI.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
  )
  var option_values = arrayof(
    [&opaque],
    [&opaque](version),
    &error_str,
    [&opaque](&error_sz)
  );

  var cx : DriverAPI.CUcontext
  var cx_created = false

  check(DriverAPI.cuCtxGetCurrent(&cx), "cuCtxGetCurrent")
  var device : CUdevice
  if cx ~= nil then
    check(DriverAPI.cuCtxGetDevice(&device), "cuCtxGetDevice")
  else
    check(DriverAPI.cuDeviceGet(&device, 0), "cuDeviceGet")
    check(DriverAPI.cuCtxCreate_v2(&cx, 0, device), "cuCtxCreate_v2")
    cx_created = true
  end

  check(DriverAPI.cuLinkCreate_v2(1, options, option_values, &linkState), "cuLinkCreate_v2")
  check(DriverAPI.cuLinkAddData_v2(linkState, DriverAPI.CU_JIT_INPUT_PTX, ptx, ptx_sz, nil, 0, nil, nil), "cuLinkAddData_v2")
  check(DriverAPI.cuLinkComplete(linkState, &cubin, &cubinSize), "cuLinkComplete")

  -- Make a copy of the returned cubin before we destroy the linker and cuda context,
  -- which may deallocate the cubin
  var to_return : rawstring = [rawstring](c.malloc(cubinSize + 1))
  to_return[cubinSize] = 0
  c.memcpy([&opaque](to_return), cubin, cubinSize)

  check(DriverAPI.cuLinkDestroy(linkState), "cuLinkDestroy")
  if cx_created then
    check(DriverAPI.cuCtxDestroy(cx), "cuCtxDestroy")
  end

  return cubin_t { to_return, cubinSize }
end

function cudahelper.jit_compile_kernels_and_register(kernels)
  local module = {}
  kernels:map(function(kernel) module[kernel.name] = kernel.kernel end)
  local version = get_cuda_version()
  local ptx = profile('cuda:ptx_gen', nil, function()
    return cudalib.toptx(module, nil, version, base.gpu_opt_profile)
  end)()

  if base.config["cuda-dump-ptx"] then io.write(ptx) end

  local cubin = nil
  local offline = base.config["offline"] or base.config["gpu-offline"]
  if not offline and base.config["cuda-generate-cubin"] then
    local ffi = require('ffi')
    cubin = profile('cuda:cubin_gen', nil, function()
      local result = ptx_to_cubin(ptx, ptx:len() + 1, version)
      return ffi.string(result.data, result.size)
    end)()
  end

  local image = cubin or ptx

  local register = quote
    var num_devices: int = -1
    checkrt(RuntimeAPI.cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount")
    escape
      for _, k in ipairs(kernels) do
        local kernel = k.kernel

        local func = kernel.cuda_func
        assert(func) -- Hopefully this is always true. If not, then it means we're generating kernels that we don't call anywhere.
        emit quote
          -- FIXME (Elliott): leaks
          func = [&DriverAPI.CUfunction](base.c.malloc(sizeof(DriverAPI.CUfunction) * num_devices))
          base.assert(func ~= nil, "allocating space for CUDA functions failed")
        end
      end
    end

    for dev_id = 0, num_devices do
      checkrt(RuntimeAPI.cudaSetDevice(dev_id), "cudaSetDevice")
      var module : DriverAPI.CUmodule
      check(DriverAPI.cuModuleLoadData(&module, image), "cuModuleLoadData")
      escape
        for _, k in ipairs(kernels) do
          local kernel = k.kernel

          local func = kernel.cuda_func
          assert(func) -- Hopefully this is always true. If not, then it means we're generating kernels that we don't call anywhere.

          emit quote
            check(DriverAPI.cuModuleGetFunction(&(func[dev_id]), module, k.name), "cuModuleGetFunction")
          end
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

cudahelper.THREAD_BLOCK_SIZE = THREAD_BLOCK_SIZE
cudahelper.NUM_THREAD_X = NUM_THREAD_X
cudahelper.NUM_THREAD_Y = NUM_THREAD_Y
cudahelper.MAX_SIZE_INLINE_KERNEL_PARAMS = MAX_SIZE_INLINE_KERNEL_PARAMS
cudahelper.GLOBAL_RED_BUFFER = GLOBAL_RED_BUFFER

local tid_x   = cudalib.nvvm_read_ptx_sreg_tid_x
local tid_y   = cudalib.nvvm_read_ptx_sreg_tid_y
local tid_z   = cudalib.nvvm_read_ptx_sreg_tid_z

local bid_x   = cudalib.nvvm_read_ptx_sreg_ctaid_x
local bid_y   = cudalib.nvvm_read_ptx_sreg_ctaid_y
local bid_z   = cudalib.nvvm_read_ptx_sreg_ctaid_z

local n_tid_x = cudalib.nvvm_read_ptx_sreg_ntid_x
local n_tid_y = cudalib.nvvm_read_ptx_sreg_ntid_y
local n_tid_z = cudalib.nvvm_read_ptx_sreg_ntid_z

local n_bid_x = cudalib.nvvm_read_ptx_sreg_nctaid_x
local n_bid_y = cudalib.nvvm_read_ptx_sreg_nctaid_y
local n_bid_z = cudalib.nvvm_read_ptx_sreg_nctaid_z

cudahelper.tid_x = tid_x
cudahelper.tid_y = tid_y
cudahelper.tid_z = tid_z
cudahelper.bid_x = bid_x
cudahelper.bid_y = bid_y
cudahelper.bid_z = bid_z

local barrier = cudalib.nvvm_barrier0

cudahelper.barrier = barrier

function cudahelper.global_thread_id()
  local bid = `(bid_x() + int64(n_bid_x()) * bid_y() + int64(n_bid_x()) * int64(n_bid_y()) * bid_z())
  local num_threads = `(n_tid_x())
  return `([bid] * [num_threads] + tid_x())
end

function cudahelper.global_thread_id_flat()
  return `(tid_x() + bid_x() * n_tid_x())
end

function cudahelper.global_block_id()
  return `(bid_x() + int64(n_bid_x()) * bid_y() + int64(n_bid_x()) * int64(n_bid_y()) * bid_z())
end

function cudahelper.sharedmemory(typ, size)
  return cudalib.sharedmemory(typ, size)
end

function cudahelper.generate_atomic_update(op, typ)
  if terralib.llvmversion <= 38 then
    if op == "+" and typ == float then
      return terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32",
                                {&float,float} -> {float})
    elseif op == "+" and typ == double and get_cuda_version() >= 60 then
      return terralib.intrinsic("llvm.nvvm.atomic.load.add.f64.p0f64",
                                {&double,double} -> {double})
    end
  end

  if typ ~= float and get_cuda_version() < 60 then
    -- For older hardware (Kepler/Maxwell), we need to generate the
    -- slow atomic since LLVM cannot generate the right instructions
    -- for these GPUs.
    local atomic_op = common.generate_slow_atomic(op, typ)
    assert(atomic_op)
    return atomic_op
  end

  local atomic_op = common.generate_atomic_update(op, typ)
  assert(atomic_op)
  return atomic_op
end

-- #####################################
-- ## Code generation for kernel launch
-- #################

function cudahelper.codegen_kernel_call(cx, kernel, count, args, shared_mem_size, tight)
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

  if not kernel.cuda_func then
    kernel.cuda_func = terralib.global(&DriverAPI.CUfunction, nil, kernel.name .. "_func")
  end
  local func = kernel.cuda_func

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
      var [stream] = RuntimeAPI.cudaGetTaskStream()
      var dev_id : int
      checkrt(RuntimeAPI.cudaGetDevice(&dev_id), "cudaGetDevice")
      [launch_domain_init]
      [setupArguments]
      [check](
        [DriverAPI.cuLaunchKernel](
          [func][dev_id],
          [grid].x, [grid].y, [grid].z,
          [block].x, [block].y, [block].z,
          [shared_mem_size], [stream], [arg_arr], nil),
        "cuLaunchKernel")
    end
  end
end

terra cudahelper.device_synchronize()
  checkrt(RuntimeAPI.cudaDeviceSynchronize())
end

local function get_nv_fn_name(name, type)
  assert(type:isfloat())
  local nv_name = "__nv_" .. name

  -- Okay. a little divergence from the C standard...
  if name == "isnan" or name == "isinf" then
    if type == double then
      nv_name = nv_name .. "d"
    else
      nv_name = nv_name .. "f"
    end
  -- Seriously?
  elseif name == "finite" then
    if type == double then
      nv_name = "__nv_isfinited"
    else
      nv_name = "__nv_finitef"
    end
  elseif type == float then
    nv_name = nv_name .. "f"
  end
  return nv_name
end

local function get_cuda_definition(self)
  if self:has_variant("cuda") then
    return self:get_variant("cuda")
  else
    local fn_type = self.super:get_definition().type
    local fn_name = get_nv_fn_name(self:get_name(), self:get_arg_type())
    assert(fn_name ~= nil)
    local fn = externcall_builtin(fn_name, fn_type)
    self:set_variant("cuda", fn)
    return fn
  end
end

function cudahelper.get_gpu_variant(math_fn)
  return math_fn:override(get_cuda_definition)
end

return cudahelper
