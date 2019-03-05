-- Copyright 2019 Stanford University, Los Alamos National Laboratory
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

local base = require("regent/std_base")
local config = require("regent/config").args()
local data = require("common/data")
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
local HijackAPI = terralib.includec("regent_cudart_hijack.h")

local C = base.c

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
      function cudahelper.check_cuda_available()
        return true
      end
    else
      function cudahelper.check_cuda_available()
        return false
      end
    end
  else
    function cudahelper.check_cuda_available()
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

local terra get_cuda_version_terra() : uint64
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
  var fat_size = sizeof(fat_bin_t)
  -- TODO: this line is leaking memory
  fat_bin = [&fat_bin_t](C.malloc(fat_size))
  base.assert(fat_size == 0 or fat_bin ~= nil, "malloc failed in register_ptx")
  fat_bin.magic = 1234
  fat_bin.versions = 5678
  var fat_data_size = ptxSize + 1
  fat_bin.data = C.malloc(fat_data_size)
  base.assert(fat_data_size == 0 or fat_bin.data ~= nil, "malloc failed in register_ptx")
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
  lua_assert(libdevice ~= nil, "Failed to find a device library")
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

local get_cuda_version
do
  local cached_cuda_version = nil
  get_cuda_version = function()
    if cached_cuda_version ~= nil then
      return cached_cuda_version
    end
    if not config["cuda-offline"] then
      cached_cuda_version = get_cuda_version_terra()
    else
      cached_cuda_version = parse_cuda_arch(config["cuda-arch"])
    end
    return cached_cuda_version
  end
end

function cudahelper.jit_compile_kernels_and_register(kernels)
  local module = {}
  for k, v in pairs(kernels) do
    module[v.name] = v.kernel
  end
  local version = get_cuda_version()
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
lua_assert(GLOBAL_RED_BUFFER % THREAD_BLOCK_SIZE == 0)

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
  local bid = `(bid_x() + n_bid_x() * bid_y() + n_bid_x() * n_bid_y() * bid_z())
  local num_threads = `(n_tid_x())
  return `([bid] * [num_threads] + tid_x())
end

function cudahelper.global_block_id()
  return `(bid_x() + n_bid_x() * bid_y() + n_bid_x() * n_bid_y() * bid_z())
end

function cudahelper.get_thread_block_size()
  return THREAD_BLOCK_SIZE
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

function cudahelper.generate_atomic_update(op, typ)
  if op == "+" and typ == float then
    return terralib.intrinsic("llvm.nvvm.atomic.load.add.f32.p0f32",
                              {&float,float} -> {float})
  elseif op == "+" and typ == double and get_cuda_version() >= 60 then
    return terralib.intrinsic("llvm.nvvm.atomic.load.add.f64.p0f64",
                              {&double,double} -> {double})
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
        new     = [base.quote_binary_op(op, assumed, operand)]
        res     = cas_func([&cas_type](address), @assumed_b, @new_b)
        old     = @[&typ](&res)
        mask    = @assumed_b == @[&cas_type](&old)
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

local internal_kernel_id = 2 ^ 30
local internal_kernels = {}
local INTERNAL_KERNEL_PREFIX = "__internal"

function cudahelper.get_internal_kernels()
  return internal_kernels
end

cudahelper.generate_buffer_init_kernel = terralib.memoize(function(type, op)
  local value = base.reduction_op_init[op][type]
  local op_name = base.reduction_ops[op].name
  local kernel_id = internal_kernel_id
  internal_kernel_id = internal_kernel_id - 1
  local kernel_name =
    INTERNAL_KERNEL_PREFIX .. "__init__" .. tostring(type) ..
    "__" .. tostring(op_name) .. "__"
  local terra init(buffer : &type)
    var tid = tid_x() + bid_x() * n_tid_x()
    buffer[tid] = [value]
  end
  init:setname(kernel_name)
  internal_kernels[kernel_id] = {
    name = kernel_name,
    kernel = init,
  }
  return kernel_id
end)

cudahelper.generate_buffer_reduction_kernel = terralib.memoize(function(type, op)
  local value = base.reduction_op_init[op][type]
  local op_name = base.reduction_ops[op].name
  local kernel_id = internal_kernel_id
  internal_kernel_id = internal_kernel_id - 1
  local kernel_name =
    INTERNAL_KERNEL_PREFIX .. "__red__" .. tostring(type) ..
    "__" .. tostring(op_name) .. "__"

  local tid = terralib.newsymbol(c.size_t, "tid")
  local input = terralib.newsymbol(&type, "input")
  local result = terralib.newsymbol(&type, "result")
  local shared_mem_ptr = cudalib.sharedmemory(type, THREAD_BLOCK_SIZE)

  local shared_mem_init = `([input][ [tid] ])
  for i = 1, (GLOBAL_RED_BUFFER / THREAD_BLOCK_SIZE) - 1 do
    shared_mem_init =
      base.quote_binary_op(op, shared_mem_init,
                           `([input][ [tid] + [i * THREAD_BLOCK_SIZE] ]))
  end
  local terra red([input], [result])
    var [tid] = tid_x()
    [shared_mem_ptr][ [tid] ] = [shared_mem_init]
    barrier()
    [cudahelper.generate_reduction_tree(tid, shared_mem_ptr, op)]
    barrier()
    if [tid] == 0 then [result][0] = [shared_mem_ptr][ [tid] ] end
  end

  red:setname(kernel_name)
  internal_kernels[kernel_id] = {
    name = kernel_name,
    kernel = red,
  }
  return kernel_id
end)

function cudahelper.generate_reduction_preamble(reductions)
  local preamble = quote end
  local device_ptrs = terralib.newlist()
  local device_ptrs_map = {}
  local host_ptrs_map = {}

  for red_var, red_op in pairs(reductions) do
    local device_ptr = terralib.newsymbol(&red_var.type, red_var.displayname)
    local host_ptr = terralib.newsymbol(&red_var.type, red_var.displayname)
    local init_kernel_id = cudahelper.generate_buffer_init_kernel(red_var.type, red_op)
    local init_args = terralib.newlist({device_ptr})
    preamble = quote
      [preamble];
      var [device_ptr] = [&red_var.type](nil)
      var [host_ptr] = [&red_var.type](nil)
      do
        var bounds : C.legion_rect_1d_t
        bounds.lo.x[0] = 0
        bounds.hi.x[0] = [sizeof(red_var.type) * GLOBAL_RED_BUFFER - 1]
        var buffer = C.legion_deferred_buffer_char_1d_create(bounds, C.GPU_FB_MEM, [&int8](nil))
        [device_ptr] =
          [&red_var.type]([&opaque](C.legion_deferred_buffer_char_1d_ptr(buffer, bounds.lo)))
        [cudahelper.codegen_kernel_call(init_kernel_id, GLOBAL_RED_BUFFER, init_args, 0, true)]
      end
      do
        var bounds : C.legion_rect_1d_t
        bounds.lo.x[0] = 0
        bounds.hi.x[0] = [sizeof(red_var.type) - 1]
        var buffer = C.legion_deferred_buffer_char_1d_create(bounds, C.Z_COPY_MEM, [&int8](nil))
        [host_ptr] =
          [&red_var.type]([&opaque](C.legion_deferred_buffer_char_1d_ptr(buffer, bounds.lo)))
      end
    end
    device_ptrs:insert(device_ptr)
    device_ptrs_map[device_ptr] = red_var
    host_ptrs_map[device_ptr] = host_ptr
  end

  return device_ptrs, device_ptrs_map, host_ptrs_map, preamble
end

function cudahelper.generate_reduction_tree(tid, shared_mem_ptr, red_op)
  local reduction_tree = quote end
  local step = THREAD_BLOCK_SIZE
  while step > 64 do
    step = step / 2
    reduction_tree = quote
      [reduction_tree]
      if [tid] < step then
        var v = [base.quote_binary_op(red_op,
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
        var v = [base.quote_binary_op(red_op,
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
  return reduction_tree
end

function cudahelper.generate_reduction_kernel(reductions, device_ptrs_map)
  local preamble = quote end
  local postamble = quote end
  for device_ptr, red_var in pairs(device_ptrs_map) do
    local red_op = reductions[red_var]
    local shared_mem_ptr =
      cudalib.sharedmemory(red_var.type, THREAD_BLOCK_SIZE)
    local init = base.reduction_op_init[red_op][red_var.type]
    preamble = quote
      [preamble]
      var [red_var] = [init]
      [shared_mem_ptr][ tid_x() ] = [red_var]
    end

    local tid = terralib.newsymbol(c.size_t, "tid")
    local reduction_tree = cudahelper.generate_reduction_tree(tid, shared_mem_ptr, red_op)
    postamble = quote
      do
        var [tid] = tid_x()
        var bid = [cudahelper.global_block_id()]
        [shared_mem_ptr][ [tid] ] = [red_var]
        barrier()
        [reduction_tree]
        if [tid] == 0 then
          [cudahelper.generate_atomic_update(red_op, red_var.type)](
            &[device_ptr][bid % [GLOBAL_RED_BUFFER] ], [shared_mem_ptr][ [tid] ])
        end
      end
    end
  end
  return preamble, postamble
end

function cudahelper.generate_reduction_postamble(reductions, device_ptrs_map, host_ptrs_map)
  local postamble = quote end
  for device_ptr, red_var in pairs(device_ptrs_map) do
    local red_op = reductions[red_var]
    local red_kernel_id = cudahelper.generate_buffer_reduction_kernel(red_var.type, red_op)
    local host_ptr = host_ptrs_map[device_ptr]
    local red_args = terralib.newlist({device_ptr, host_ptr})
    local shared_mem_size = terralib.sizeof(red_var.type) * THREAD_BLOCK_SIZE
    postamble = quote
      [postamble];
      [cudahelper.codegen_kernel_call(red_kernel_id, THREAD_BLOCK_SIZE, red_args, shared_mem_size, true)]
    end
  end

  local needs_sync = true
  for device_ptr, red_var in pairs(device_ptrs_map) do
    if needs_sync then
      postamble = quote
        [postamble];
        RuntimeAPI.cudaDeviceSynchronize()
      end
      needs_sync = false
    end
    local red_op = reductions[red_var]
    local host_ptr = host_ptrs_map[device_ptr]
    postamble = quote
      [postamble];
      [red_var] = [base.quote_binary_op(red_op, red_var, `([host_ptr][0]))]
    end
  end

  return postamble
end

-- #####################################
-- ## Code generation for parallel prefix operators
-- #################

local NUM_BANKS = 16
local bank_offset = macro(function(e)
  return `(e / [NUM_BANKS])
end)

local function generate_prefix_op_kernel(shmem, tid, num_leaves, op, init, left_to_right)
  return quote
    do
      var oa = 2 * [tid] + 1
      var ob = 2 * [tid] + 2 * [left_to_right]
      var d : int = [num_leaves] >> 1
      var offset = 1
      while d > 0 do
        barrier()
        if [tid]  < d then
          var ai : int = offset * oa - [left_to_right]
          var bi : int = offset * ob - [left_to_right]
          ai = ai + bank_offset(ai)
          bi = bi + bank_offset(bi)
          [shmem][bi] = [base.quote_binary_op(op, `([shmem][ai]), `([shmem][bi]))]
        end
        offset = offset << 1
        d = d >> 1
      end
      if [tid] == 0 then
        var idx = ([num_leaves] - [left_to_right]) % [num_leaves]
        [shmem][idx + bank_offset(idx)] = [init]
      end
      d = 1
      while d <= [num_leaves] do
        offset = offset >> 1
        barrier()
        if [tid] < d and offset > 0 then
          var ai = offset * oa - [left_to_right]
          var bi = offset * ob - [left_to_right]
          ai = ai + bank_offset(ai)
          bi = bi + bank_offset(bi)
          var x = [shmem][ai]
          [shmem][ai] = [shmem][bi]
          [shmem][bi] = [base.quote_binary_op(op, x, `([shmem][bi]))]
        end
        d = d << 1
      end
      barrier()
    end
  end
end

local function generate_prefix_op_prescan(shmem, lhs, rhs, lhs_ptr, rhs_ptr, res, idx, dir, op, init)
  local prescan_full, prescan_arbitrary
  local NUM_LEAVES = THREAD_BLOCK_SIZE * 2

  local function advance_ptrs(lhs_ptr, rhs_ptr, bid)
    if lhs_ptr == rhs_ptr then
      return quote
        [lhs_ptr] = &([lhs_ptr][ bid * [NUM_LEAVES] ])
      end
    else
      return quote
        [lhs_ptr] = &([lhs_ptr][ bid * [NUM_LEAVES] ])
        [rhs_ptr] = &([rhs_ptr][ bid * [NUM_LEAVES] ])
      end
    end
  end

  terra prescan_full([lhs_ptr],
                     [rhs_ptr],
                     [dir])
    var [idx]
    var t = tid_x()
    var bid = [cudahelper.global_block_id()]
    [advance_ptrs(lhs_ptr, rhs_ptr, bid)]
    var lr = [int]([dir] >= 0)

    [idx].__ptr = t
    [rhs.actions]
    [shmem][ [idx].__ptr + bank_offset([idx].__ptr)] = [rhs.value]
    [idx].__ptr = [idx].__ptr + [THREAD_BLOCK_SIZE]
    [rhs.actions]
    [shmem][ [idx].__ptr + bank_offset([idx].__ptr)] = [rhs.value]

    [generate_prefix_op_kernel(shmem, t, NUM_LEAVES, op, init, lr)]

    var [res]
    [idx].__ptr = t
    [rhs.actions]
    [res] = [base.quote_binary_op(op, `([shmem][ [idx].__ptr + bank_offset([idx].__ptr) ]), rhs.value)]
    [lhs.actions]
    [idx].__ptr = [idx].__ptr + [THREAD_BLOCK_SIZE]
    [rhs.actions]
    [res] = [base.quote_binary_op(op, `([shmem][ [idx].__ptr + bank_offset([idx].__ptr) ]), rhs.value)]
    [lhs.actions]
  end

  terra prescan_arbitrary([lhs_ptr],
                          [rhs_ptr],
                          num_elmts : c.size_t,
                          num_leaves : c.size_t,
                          [dir])
    var [idx]
    var t = tid_x()
    var lr = [int]([dir] >= 0)

    [idx].__ptr = t
    [rhs.actions]
    [shmem][ [idx].__ptr + bank_offset([idx].__ptr)] = [rhs.value]
    [idx].__ptr = [idx].__ptr + (num_leaves / 2)
    if [idx].__ptr < num_elmts then
      [rhs.actions]
      [shmem][ [idx].__ptr + bank_offset([idx].__ptr) ] = [rhs.value]
    else
      [shmem][ [idx].__ptr + bank_offset([idx].__ptr) ] = [init]
    end

    [generate_prefix_op_kernel(shmem, t, num_leaves, op, init, lr)]

    var [res]
    [idx].__ptr = t
    [rhs.actions]
    [res] = [base.quote_binary_op(op,
        `([shmem][ [idx].__ptr + bank_offset([idx].__ptr) ]), rhs.value)]
    [lhs.actions]
    [idx].__ptr = [idx].__ptr + (num_leaves / 2)
    if [idx].__ptr < num_elmts then
      [rhs.actions]
      [res] = [base.quote_binary_op(op,
          `([shmem][ [idx].__ptr + bank_offset([idx].__ptr) ]), rhs.value)]
      [lhs.actions]
    end
  end

  return prescan_full, prescan_arbitrary
end

local function generate_prefix_op_scan(shmem, lhs_wr, lhs_rd, lhs_ptr, res, idx, dir, op, init)
  local scan_full, scan_arbitrary
  local NUM_LEAVES = THREAD_BLOCK_SIZE * 2

  terra scan_full([lhs_ptr],
                  offset : uint64,
                  [dir])
    var [idx]
    var t = tid_x()
    var bid = [cudahelper.global_block_id()]
    [lhs_ptr] = &([lhs_ptr][ bid * [offset] * [NUM_LEAVES] ])
    var lr = [int]([dir] >= 0)

    var tidx = t
    [idx].__ptr = (tidx + lr) * [offset] - lr
    [lhs_rd.actions]
    [shmem][tidx + bank_offset(tidx)] = [lhs_rd.value]
    tidx = tidx + [THREAD_BLOCK_SIZE]
    [idx].__ptr = (tidx + lr) * [offset] - lr
    [lhs_rd.actions]
    [shmem][tidx + bank_offset(tidx)] = [lhs_rd.value]

    [generate_prefix_op_kernel(shmem, t, NUM_LEAVES, op, init, lr)]

    var [res]
    tidx = t
    [idx].__ptr = (tidx + lr) * [offset] - lr
    [lhs_rd.actions]
    [res] = [base.quote_binary_op(op, `([shmem][tidx + bank_offset(tidx)]), lhs_rd.value)]
    [lhs_wr.actions]
    tidx = tidx + [THREAD_BLOCK_SIZE]
    [idx].__ptr = (tidx + lr) * [offset] - lr
    [lhs_rd.actions]
    [res] = [base.quote_binary_op(op, `([shmem][tidx + bank_offset(tidx)]), lhs_rd.value)]
    [lhs_wr.actions]
  end

  terra scan_arbitrary([lhs_ptr],
                       num_elmts : c.size_t,
                       num_leaves : c.size_t,
                       offset : c.size_t,
                       [dir])
    var [idx]
    var t = tid_x()
    var lr = [int]([dir] >= 0)

    if lr == 1 then
      var tidx = t
      [idx].__ptr = (tidx + 1) * [offset] - 1
      [lhs_rd.actions]
      [shmem][tidx + bank_offset(tidx)] = [lhs_rd.value]

      tidx = t + num_leaves / 2
      if tidx < [num_elmts] then
        [idx].__ptr = (tidx + 1) * [offset] - 1
        [lhs_rd.actions]
        [shmem][tidx + bank_offset(tidx)] = [lhs_rd.value]
      else
        [shmem][tidx + bank_offset(tidx)] = [init]
      end
    else
      var tidx = t
      [idx].__ptr = tidx * [offset]
      [lhs_rd.actions]
      [shmem][tidx + bank_offset(tidx)] = [lhs_rd.value]
      tidx = t + num_leaves / 2
      if tidx < [num_elmts] then
        [idx].__ptr = tidx * [offset]
        [lhs_rd.actions]
        [shmem][tidx + bank_offset(tidx)] = [lhs_rd.value]
      else
        [shmem][tidx + bank_offset(tidx)] = [init]
      end
    end

    [generate_prefix_op_kernel(shmem, t, num_leaves, op, init, lr)]

    var [res]
    if lr == 1 then
      var tidx = t
      [idx].__ptr = (tidx + 1) * [offset] - 1
      [lhs_rd.actions]
      [res] = [base.quote_binary_op(op, `([shmem][tidx + bank_offset(tidx)]), lhs_rd.value)]
      [lhs_wr.actions]
      tidx = tidx + num_leaves / 2
      if [tidx] < [num_elmts] then
        [idx].__ptr = (tidx + 1) * [offset] - 1
        [lhs_rd.actions]
        [res] = [base.quote_binary_op(op, `([shmem][tidx + bank_offset(tidx)]), lhs_rd.value)]
        [lhs_wr.actions]
      end
    else
      var tidx = t
      [idx].__ptr = tidx * [offset]
      [lhs_rd.actions]
      [res] = [base.quote_binary_op(op, `([shmem][tidx + bank_offset(tidx)]), lhs_rd.value)]
      [lhs_wr.actions]
      tidx = tidx + num_leaves / 2
      if [tidx] < [num_elmts] then
        [idx].__ptr = tidx * [offset]
        [lhs_rd.actions]
        [res] = [base.quote_binary_op(op, `([shmem][tidx + bank_offset(tidx)]), lhs_rd.value)]
        [lhs_wr.actions]
      end
    end
  end

  return scan_full, scan_arbitrary
end

-- This function expects lhs and rhs to be the values from the following expressions.
--
--   * lhs: lhs[idx] = res
--   * rhs: rhs[idx]
--
-- The code generator below captures 'idx' and 'res' to change the meaning of these values
function cudahelper.generate_prefix_op_kernels(lhs_wr, lhs_rd, rhs, lhs_ptr, rhs_ptr,
                                               res, idx, dir, op, elem_type)
  local BLOCK_SIZE = THREAD_BLOCK_SIZE * 2
  local shmem = cudalib.sharedmemory(elem_type, BLOCK_SIZE)
  local init = base.reduction_op_init[op][elem_type]

  local prescan_full, prescan_arbitrary =
    generate_prefix_op_prescan(shmem, lhs_wr, rhs, lhs_ptr, rhs_ptr, res, idx, dir, op, init)

  local scan_full, scan_arbitrary =
    generate_prefix_op_scan(shmem, lhs_wr, lhs_rd, lhs_ptr, res, idx, dir, op, init)

  local terra postscan_full([lhs_ptr],
                            offset : uint64,
                            num_elmts : uint64,
                            [dir])
    var t = [cudahelper.global_thread_id()]
    if t >= num_elmts - [BLOCK_SIZE] or t % [BLOCK_SIZE] == [BLOCK_SIZE - 1]then return end

    var sum_loc = t / [BLOCK_SIZE] * [BLOCK_SIZE] + [BLOCK_SIZE - 1]
    var val_loc = t + [BLOCK_SIZE]
    var [idx], [res]
    if [dir] >= 0 then
      [idx].__ptr = sum_loc * [offset] + ([offset] - 1)
      [lhs_rd.actions]
      var v1 = [lhs_rd.value]

      [idx].__ptr = val_loc * [offset] + ([offset] - 1)
      [lhs_rd.actions]
      var v2 = [lhs_rd.value]

      [res] = [base.quote_binary_op(op, v1, v2)]
      [lhs_wr.actions]
    else
      var t = [cudahelper.global_thread_id()]
      if t % [BLOCK_SIZE] == [BLOCK_SIZE - 1] then return end

      [idx].__ptr = (num_elmts - 1 - sum_loc) * [offset]
      [lhs_rd.actions]
      var v1 = [lhs_rd.value]

      [idx].__ptr = (num_elmts - 1 - val_loc) * [offset]
      [lhs_rd.actions]
      var v2 = [lhs_rd.value]

      [res] = [base.quote_binary_op(op, v1, v2)]
      [lhs_wr.actions]
    end
  end

  return prescan_full, prescan_arbitrary, scan_full, scan_arbitrary, postscan_full
end

function cudahelper.generate_parallel_prefix_op(variant, total, lhs_wr, lhs_rd, rhs, lhs_ptr, rhs_ptr,
                                                res, idx, dir, op, elem_type)
  local BLOCK_SIZE = THREAD_BLOCK_SIZE * 2
  local SHMEM_SIZE = terralib.sizeof(elem_type) * THREAD_BLOCK_SIZE * 2

  local pre_full, pre_arb, scan_full, scan_arb, post_full, post2, post3 =
    cudahelper.generate_prefix_op_kernels(lhs_wr, lhs_rd, rhs, lhs_ptr, rhs_ptr,
                                          res, idx, dir, op, elem_type)
  local prescan_full_id = variant:add_cuda_kernel(pre_full)
  local prescan_arb_id = variant:add_cuda_kernel(pre_arb)
  local scan_full_id = variant:add_cuda_kernel(scan_full)
  local scan_arb_id = variant:add_cuda_kernel(scan_arb)
  local postscan_full_id = variant:add_cuda_kernel(post_full)

  local num_leaves = terralib.newsymbol(c.size_t, "num_leaves")
  local num_elmts = terralib.newsymbol(c.size_t, "num_elmts")
  local num_threads = terralib.newsymbol(c.size_t, "num_threads")
  local offset = terralib.newsymbol(uint64, "offset")
  local lhs_ptr_arg = terralib.newsymbol(lhs_ptr.type, lhs_ptr.name)
  local rhs_ptr_arg = terralib.newsymbol(rhs_ptr.type, rhs_ptr.name)

  local prescan_full_args = terralib.newlist()
  prescan_full_args:insertall({lhs_ptr_arg, rhs_ptr_arg, dir})
  local call_prescan_full =
    cudahelper.codegen_kernel_call(prescan_full_id, num_threads, prescan_full_args, SHMEM_SIZE, true)

  local prescan_arb_args = terralib.newlist()
  prescan_arb_args:insertall({lhs_ptr_arg, rhs_ptr_arg, num_elmts, num_leaves, dir})
  local call_prescan_arbitrary =
    cudahelper.codegen_kernel_call(prescan_arb_id, num_threads, prescan_arb_args, SHMEM_SIZE, true)

  local scan_full_args = terralib.newlist()
  scan_full_args:insertall({lhs_ptr_arg, offset, dir})
  local call_scan_full =
    cudahelper.codegen_kernel_call(scan_full_id, num_threads, scan_full_args, SHMEM_SIZE, true)

  local scan_arb_args = terralib.newlist()
  scan_arb_args:insertall({lhs_ptr_arg, num_elmts, num_leaves, offset, dir})
  local call_scan_arbitrary =
    cudahelper.codegen_kernel_call(scan_arb_id, num_threads, scan_arb_args, SHMEM_SIZE, true)

  local postscan_full_args = terralib.newlist()
  postscan_full_args:insertall({lhs_ptr, offset, num_elmts, dir})
  local call_postscan_full =
    cudahelper.codegen_kernel_call(postscan_full_id, num_threads, postscan_full_args, 0, true)

  local terra recursive_scan :: {uint64,uint64,uint64,lhs_ptr.type,dir.type} -> {}

  terra recursive_scan(remaining : uint64,
                       [offset],
                       [total],
                       [lhs_ptr],
                       [dir])
    if remaining <= 1 then return end

    var num_blocks : uint64 = remaining / [BLOCK_SIZE]

    if num_blocks > 0 then
      var [num_threads] = num_blocks * [THREAD_BLOCK_SIZE]
      var [lhs_ptr_arg]
      if [dir] >= 0 then
        [lhs_ptr_arg] = [lhs_ptr]
      else
        [lhs_ptr_arg] = &[lhs_ptr][(remaining % [BLOCK_SIZE]) * [offset]]
      end
      [call_scan_full]
    end
    if remaining % [BLOCK_SIZE] > 0 then
      var [lhs_ptr_arg]
      if [dir] >= 0 then
        [lhs_ptr_arg] = &[lhs_ptr][ num_blocks * [BLOCK_SIZE] * [offset] ]
      else
        [lhs_ptr_arg] = [lhs_ptr]
      end
      var [num_elmts] = remaining % [BLOCK_SIZE]
      var [num_leaves] = [BLOCK_SIZE]
      while [num_leaves] / 2 > [num_elmts] do
        [num_leaves] = [num_leaves] / 2
      end
      var [num_threads] = [num_leaves] / 2
      [call_scan_arbitrary]
    end

    var [lhs_ptr_arg]
    if [dir] >= 0 then
      [lhs_ptr_arg] = [lhs_ptr]
    else
      [lhs_ptr_arg] = &[lhs_ptr][ (remaining % [BLOCK_SIZE]) * [offset] ]
    end

    recursive_scan(num_blocks,
                   [offset] * [BLOCK_SIZE],
                   [total],
                   [lhs_ptr_arg],
                   [dir])

    if [remaining] > [BLOCK_SIZE] then
      var [num_elmts] = remaining
      var [num_threads] = remaining - [BLOCK_SIZE]
      [call_postscan_full]
    end
  end


  local launch = quote
    do
      var num_blocks : uint64 = total / [BLOCK_SIZE]
      if num_blocks > 0 then
        var [lhs_ptr_arg]
        var [rhs_ptr_arg]
        var [num_threads] = num_blocks * [THREAD_BLOCK_SIZE]
        if [dir] >= 0 then
          [lhs_ptr_arg] = [lhs_ptr]
          [rhs_ptr_arg] = [rhs_ptr]
        else
          [lhs_ptr_arg] = &[lhs_ptr][ total % [BLOCK_SIZE] ]
          [rhs_ptr_arg] = &[rhs_ptr][ total % [BLOCK_SIZE] ]
        end
        [call_prescan_full]
      end
      if total % [BLOCK_SIZE] > 0 then
        var [lhs_ptr_arg]
        var [rhs_ptr_arg]
        if [dir] >= 0 then
          [lhs_ptr_arg] = &[lhs_ptr][ num_blocks * [BLOCK_SIZE] ]
          [rhs_ptr_arg] = &[rhs_ptr][ num_blocks * [BLOCK_SIZE] ]
        else
          [lhs_ptr_arg] = [lhs_ptr]
          [rhs_ptr_arg] = [rhs_ptr]
        end
        var [num_elmts] = total % [BLOCK_SIZE]
        var [num_leaves] = [BLOCK_SIZE]
        while [num_leaves] / 2 > [num_elmts] do
          [num_leaves] = [num_leaves] / 2
        end
        var [num_threads] = [num_leaves] / 2
        [call_prescan_arbitrary]
      end

      var [lhs_ptr_arg]
      if [dir] >= 0 then
        [lhs_ptr_arg] = [lhs_ptr]
      else
        [lhs_ptr_arg] = &[lhs_ptr][ total % [BLOCK_SIZE] ]
      end

      recursive_scan(total / [BLOCK_SIZE],
                     [BLOCK_SIZE],
                     [total],
                     [lhs_ptr_arg],
                     [dir])

      if total > [BLOCK_SIZE] then
        var [offset] = 1
        var [num_elmts] = total
        var [num_threads] = total - [BLOCK_SIZE]
        [call_postscan_full]
      end
    end
  end

  return launch
end

function cudahelper.codegen_kernel_call(kernel_id, count, args, shared_mem_size, tight)
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
    if [count] <= THREAD_BLOCK_SIZE and tight then
      [block].x, [block].y, [block].z = [count], 1, 1
    else
      [block].x, [block].y, [block].z = THREAD_BLOCK_SIZE, 1, 1
    end
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

local function get_nv_fn_name(name, type)
  lua_assert(type:isfloat())
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
    lua_assert(fn_name ~= nil)
    local fn = externcall_builtin(fn_name, fn_type)
    self:set_variant("cuda", fn)
    return fn
  end
end

function cudahelper.get_cuda_variant(math_fn)
  return math_fn:override(get_cuda_definition)
end

return cudahelper
