-- Copyright 2022 Stanford University, Los Alamos National Laboratory
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

-- Regent GPU Helper Code
--
-- This file contains helper code for the Regent GPU code generator.
--
-- IMPORTANT: the GPU implementation is broken into multiple files
-- that contain DIFFERENT types of code.
--
--  * This file contains DEVICE INDEPENDENT code that depends on
--    DEVICE-SPECIFIC configuration settings.
--
--  * gpu/cuda.t, gpu/hip.t: contain DEVICE-SPECIFIC code for a given
--    platform. Ideally this should be minimized, because the goal is
--    to share code between platforms.
--
--  * gpu/common.t: contains DEVICE INDEPENDENT code that does not
--    depend on any device-specific settings.

local ast = require("regent/ast")
local base = require("regent/std_base")
local data = require("common/data")

local gpuhelper = {}

local config = base.config
local c = base.c

local function is_disabled_value(v)
  return v == "none" or v == "off" or v == ""
end
local function is_unspecified_value(v)
  return v == "unspecified"
end

local requested = not (is_unspecified_value(config["gpu"]) or is_disabled_value(config["gpu"]))
function gpuhelper.is_gpu_requested()
  return requested
end

local reason
if is_disabled_value(config["gpu"]) then
  reason = "GPU support was disabled on the command line"
elseif is_unspecified_value(config["gpu"]) then
  -- If unspecified, try to detect CUDA first, then fall back to off (for now).
  local cudaimpl = require("regent/gpu/cuda")
  local available, cuda_reason = cudaimpl.check_gpu_available()
  if available then
    config["gpu"] = "cuda"
  else
    config["gpu"] = "none"
    reason = "GPU auto-detect failed: " .. tostring(cuda_reason)
  end
end

-- Exit early if the user turned off GPU code generation (or auto-detect failed).
local impl
if is_disabled_value(config["gpu"]) then
  function gpuhelper.check_gpu_available()
    return false, reason
  end
  return gpuhelper
elseif config["gpu"] == "cuda" then
  impl = require("regent/gpu/cuda")
elseif config["gpu"] == "hip" then
  impl = require("regent/gpu/hip")
else
  assert(false)
end

gpuhelper.check_gpu_available = impl.check_gpu_available

do
  local available, error_message = gpuhelper.check_gpu_available()
  if not available then
    if requested then
      print("GPU code generation failed since " .. error_message)
      os.exit(-1)
    else
      return gpuhelper
    end
  end
end

gpuhelper.link_driver_library = impl.link_driver_library

gpuhelper.driver_library_link_flags = impl.driver_library_link_flags

gpuhelper.jit_compile_kernels_and_register = impl.jit_compile_kernels_and_register

gpuhelper.global_thread_id = impl.global_thread_id

gpuhelper.generate_atomic_update = impl.generate_atomic_update

gpuhelper.get_gpu_variant = impl.get_gpu_variant

gpuhelper.codegen_kernel_call = impl.codegen_kernel_call

-- #####################################
-- ## Primitives
-- #################

local THREAD_BLOCK_SIZE = impl.THREAD_BLOCK_SIZE
local NUM_THREAD_X = impl.NUM_THREAD_X
local NUM_THREAD_Y = impl.NUM_THREAD_Y
local MAX_SIZE_INLINE_KERNEL_PARAMS = impl.MAX_SIZE_INLINE_KERNEL_PARAMS
local GLOBAL_RED_BUFFER = impl.GLOBAL_RED_BUFFER

local tid_x = impl.tid_x
local tid_y = impl.tid_y
local tid_z = impl.tid_z
local bid_x = impl.bid_x
local bid_y = impl.bid_y
local bid_z = impl.bid_z

local barrier = impl.barrier

-- #####################################
-- ## Code generation for scalar reduction
-- #################

local supported_scalar_red_ops = {
  ["+"]   = true,
  ["*"]   = true,
  ["max"] = true,
  ["min"] = true,
}

function gpuhelper.compute_reduction_buffer_size(cx, node, reductions)
  local size = 0
  for k, v in pairs(reductions) do
    if not supported_scalar_red_ops[v] then
      report.error(node,
          "Scalar reduction with operator " .. v .. " is not supported yet")
    elseif not (sizeof(k.type) == 4 or sizeof(k.type) == 8) then
      report.error(node,
          "Scalar reduction for type " .. tostring(k.type) .. " is not supported yet")
    end
    size = size + THREAD_BLOCK_SIZE * sizeof(k.type)
  end
  size = size + cx:compute_reduction_buffer_size()
  return size
end

local internal_kernels = terralib.newlist()
local INTERNAL_KERNEL_PREFIX = "__internal"

function gpuhelper.get_internal_kernels()
  return internal_kernels
end

gpuhelper.generate_buffer_init_kernel = terralib.memoize(function(type, op)
  local value = base.reduction_op_init[op][type]
  local op_name = base.reduction_ops[op].name
  local kernel_name =
    INTERNAL_KERNEL_PREFIX .. "__init__" .. tostring(type) ..
    "__" .. tostring(op_name) .. "__"
  local terra init(buffer : &type)
    var tid = [impl.global_thread_id_flat()]
    buffer[tid] = [value]
  end
  init:setname(kernel_name)
  internal_kernels:insert({
    name = kernel_name,
    kernel = init,
  })
  return init
end)

gpuhelper.generate_buffer_reduction_kernel = terralib.memoize(function(type, op)
  local value = base.reduction_op_init[op][type]
  local op_name = base.reduction_ops[op].name
  local kernel_name =
    INTERNAL_KERNEL_PREFIX .. "__red__" .. tostring(type) ..
    "__" .. tostring(op_name) .. "__"

  local tid = terralib.newsymbol(c.size_t, "tid")
  local input = terralib.newsymbol(&type, "input")
  local result = terralib.newsymbol(&type, "result")
  local shared_mem_ptr = impl.sharedmemory(type, THREAD_BLOCK_SIZE)

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
    [gpuhelper.generate_reduction_tree(tid, shared_mem_ptr, THREAD_BLOCK_SIZE, op, type)]
    barrier()
    if [tid] == 0 then [result][0] = [shared_mem_ptr][ [tid] ] end
  end

  red:setname(kernel_name)
  internal_kernels:insert({
    name = kernel_name,
    kernel = red,
  })
  return red
end)

function gpuhelper.generate_reduction_preamble(cx, reductions)
  local preamble = terralib.newlist()
  local device_ptrs = terralib.newlist()
  local buffer_cleanups = terralib.newlist()
  local device_ptrs_map = {}
  local host_ptrs_map = {}

  for red_var, red_op in pairs(reductions) do
    local device_ptr = terralib.newsymbol(&red_var.type, red_var.displayname)
    local host_ptr = terralib.newsymbol(&red_var.type, red_var.displayname)
    local device_buffer =
      terralib.newsymbol(c.legion_deferred_buffer_char_1d_t,
                         "__d_buffer_" .. red_var.displayname)
    local host_buffer =
      terralib.newsymbol(c.legion_deferred_buffer_char_1d_t,
                         "__h_buffer_" .. red_var.displayname)
    local init_kernel = gpuhelper.generate_buffer_init_kernel(red_var.type, red_op)
    local init_args = terralib.newlist({device_ptr})
    preamble:insert(quote
      var [device_ptr] = [&red_var.type](nil)
      var [host_ptr] = [&red_var.type](nil)
      var [device_buffer]
      var [host_buffer]
      do
        var bounds : c.legion_rect_1d_t
        bounds.lo.x[0] = 0
        bounds.hi.x[0] = [sizeof(red_var.type) * GLOBAL_RED_BUFFER - 1]
        [device_buffer] = c.legion_deferred_buffer_char_1d_create(bounds, c.GPU_FB_MEM, [&int8](nil))
        [device_ptr] =
          [&red_var.type]([&opaque](c.legion_deferred_buffer_char_1d_ptr([device_buffer], bounds.lo)))
        [gpuhelper.codegen_kernel_call(cx, init_kernel, GLOBAL_RED_BUFFER, init_args, 0, true)]
      end
      do
        var bounds : c.legion_rect_1d_t
        bounds.lo.x[0] = 0
        bounds.hi.x[0] = [sizeof(red_var.type) - 1]
        [host_buffer] = c.legion_deferred_buffer_char_1d_create(bounds, c.Z_COPY_MEM, [&int8](nil))
        [host_ptr] =
          [&red_var.type]([&opaque](c.legion_deferred_buffer_char_1d_ptr([host_buffer], bounds.lo)))
      end
    end)
    device_ptrs:insert(device_ptr)
    device_ptrs_map[device_ptr] = red_var
    host_ptrs_map[device_ptr] = host_ptr
    buffer_cleanups:insert(quote
      c.legion_deferred_buffer_char_1d_destroy([host_buffer])
      c.legion_deferred_buffer_char_1d_destroy([device_buffer])
    end)
  end

  return device_ptrs, device_ptrs_map, host_ptrs_map, preamble, buffer_cleanups
end

local function generate_element_reduction(lhs, rhs, op, volatile)
  if volatile then
    return quote
      do
        var v = [base.quote_binary_op(op, lhs, rhs)]
        terralib.attrstore(&[lhs], v, { isvolatile = true })
      end
    end
  else
    return quote
      [lhs] = [base.quote_binary_op(op, lhs, rhs)]
    end
  end
end

local function generate_element_reductions(lhs, rhs, op, type, volatile)
  local actions = terralib.newlist()
  if type:isarray() then
    for k = 1, type.N do -- inclusive!
      local lhs = `([lhs][ [k - 1] ])
      local rhs = `([rhs][ [k - 1] ])
      actions:insert(generate_element_reduction(lhs, rhs, op, volatile))
    end
  else
    assert(type:isprimitive())
    actions:insert(generate_element_reduction(lhs, rhs, op, volatile))
  end
  return quote [actions] end
end

function gpuhelper.generate_reduction_tree(tid, shared_mem_ptr, num_threads, red_op, type)
  local outer_reductions = terralib.newlist()
  local step = num_threads
  while step > 64 do
    step = step / 2
    outer_reductions:insert(quote
      if [tid] < step then
        [generate_element_reductions(`([shared_mem_ptr][ [tid] ]),
                                     `([shared_mem_ptr][ [tid] + [step] ]),
                                     red_op, type, false)]
      end
      barrier()
    end)
  end
  local unrolled_reductions = terralib.newlist()
  while step > 1 do
    step = step / 2
    unrolled_reductions:insert(quote
      if [tid] < [step] then
        [generate_element_reductions(`([shared_mem_ptr][ [tid] ]),
                                     `([shared_mem_ptr][ [tid] + [step] ]),
                                     red_op, type, false)]
      end
      barrier()
    end)
  end
  if #outer_reductions > 0 then
    return quote
      [outer_reductions]
      if [tid] < 32 then
        [unrolled_reductions]
      end
    end
  else
    return quote
      [unrolled_reductions]
    end
  end
end

function gpuhelper.generate_reduction_kernel(cx, reductions, device_ptrs_map)
  local preamble = terralib.newlist()
  local postamble = terralib.newlist()
  for device_ptr, red_var in pairs(device_ptrs_map) do
    local red_op = reductions[red_var]
    local shared_mem_ptr =
      impl.sharedmemory(red_var.type, THREAD_BLOCK_SIZE)
    local init = base.reduction_op_init[red_op][red_var.type]
    preamble:insert(quote
      var [red_var] = [init]
      [shared_mem_ptr][ tid_x() ] = [red_var]
    end)

    local tid = terralib.newsymbol(c.size_t, "tid")
    local reduction_tree =
      gpuhelper.generate_reduction_tree(tid, shared_mem_ptr, THREAD_BLOCK_SIZE, red_op, red_var.type)
    postamble:insert(quote
      do
        var [tid] = tid_x()
        var bid = [impl.global_block_id()]
        [shared_mem_ptr][ [tid] ] = [red_var]
        barrier()
        [reduction_tree]
        if [tid] == 0 then
          [gpuhelper.generate_atomic_update(red_op, red_var.type)](
            &[device_ptr][bid % [GLOBAL_RED_BUFFER] ], [shared_mem_ptr][ [tid] ])
        end
      end
    end)
  end

  preamble:insertall(cx:generate_preamble())
  postamble:insertall(cx:generate_postamble())

  return preamble, postamble
end

function gpuhelper.generate_reduction_postamble(cx, reductions, device_ptrs_map, host_ptrs_map)
  local postamble = quote end
  for device_ptr, red_var in pairs(device_ptrs_map) do
    local red_op = reductions[red_var]
    local red_kernel_name = gpuhelper.generate_buffer_reduction_kernel(red_var.type, red_op)
    local host_ptr = host_ptrs_map[device_ptr]
    local red_args = terralib.newlist({device_ptr, host_ptr})
    local shared_mem_size = terralib.sizeof(red_var.type) * THREAD_BLOCK_SIZE
    postamble = quote
      [postamble];
      [gpuhelper.codegen_kernel_call(cx, red_kernel_name, THREAD_BLOCK_SIZE, red_args, shared_mem_size, true)]
    end
  end

  local needs_sync = true
  for device_ptr, red_var in pairs(device_ptrs_map) do
    if needs_sync then
      postamble = quote
        [postamble];
        impl.device_synchronize()
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
    var bid = [impl.global_block_id()]
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
    var bid = [impl.global_block_id()]
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
function gpuhelper.generate_prefix_op_kernels(lhs_wr, lhs_rd, rhs, lhs_ptr, rhs_ptr,
                                               res, idx, dir, op, elem_type)
  local BLOCK_SIZE = THREAD_BLOCK_SIZE * 2
  local shmem = impl.sharedmemory(elem_type, BLOCK_SIZE)
  local init = base.reduction_op_init[op][elem_type]

  local prescan_full, prescan_arbitrary =
    generate_prefix_op_prescan(shmem, lhs_wr, rhs, lhs_ptr, rhs_ptr, res, idx, dir, op, init)

  local scan_full, scan_arbitrary =
    generate_prefix_op_scan(shmem, lhs_wr, lhs_rd, lhs_ptr, res, idx, dir, op, init)

  local terra postscan_full([lhs_ptr],
                            offset : uint64,
                            num_elmts : uint64,
                            [dir])
    var t = [gpuhelper.global_thread_id()]
    if t >= num_elmts - [BLOCK_SIZE] or t % [BLOCK_SIZE] == [BLOCK_SIZE - 1] then return end

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
      var t = [gpuhelper.global_thread_id()]
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

function gpuhelper.generate_parallel_prefix_op(cx, variant, total, lhs_wr, lhs_rd, rhs, lhs_ptr,
                                                rhs_ptr, res, idx, dir, op, elem_type)
  local BLOCK_SIZE = THREAD_BLOCK_SIZE * 2
  local SHMEM_SIZE = terralib.sizeof(elem_type) * THREAD_BLOCK_SIZE * 2

  local pre_full, pre_arb, scan_full, scan_arb, post_full, post2, post3 =
    gpuhelper.generate_prefix_op_kernels(lhs_wr, lhs_rd, rhs, lhs_ptr, rhs_ptr,
                                          res, idx, dir, op, elem_type)
  variant:add_cuda_kernel(pre_full)
  variant:add_cuda_kernel(pre_arb)
  variant:add_cuda_kernel(scan_full)
  variant:add_cuda_kernel(scan_arb)
  variant:add_cuda_kernel(post_full)

  local num_leaves = terralib.newsymbol(c.size_t, "num_leaves")
  local num_elmts = terralib.newsymbol(c.size_t, "num_elmts")
  local num_threads = terralib.newsymbol(c.size_t, "num_threads")
  local offset = terralib.newsymbol(uint64, "offset")
  local lhs_ptr_arg = terralib.newsymbol(lhs_ptr.type, lhs_ptr.name)
  local rhs_ptr_arg = terralib.newsymbol(rhs_ptr.type, rhs_ptr.name)

  local prescan_full_args = terralib.newlist()
  prescan_full_args:insertall({lhs_ptr_arg, rhs_ptr_arg, dir})
  local call_prescan_full =
    gpuhelper.codegen_kernel_call(cx, pre_full, num_threads, prescan_full_args, SHMEM_SIZE, true)

  local prescan_arb_args = terralib.newlist()
  prescan_arb_args:insertall({lhs_ptr_arg, rhs_ptr_arg, num_elmts, num_leaves, dir})
  local call_prescan_arbitrary =
    gpuhelper.codegen_kernel_call(cx, pre_arb, num_threads, prescan_arb_args, SHMEM_SIZE, true)

  local scan_full_args = terralib.newlist()
  scan_full_args:insertall({lhs_ptr_arg, offset, dir})
  local call_scan_full =
    gpuhelper.codegen_kernel_call(cx, scan_full, num_threads, scan_full_args, SHMEM_SIZE, true)

  local scan_arb_args = terralib.newlist()
  scan_arb_args:insertall({lhs_ptr_arg, num_elmts, num_leaves, offset, dir})
  local call_scan_arbitrary =
    gpuhelper.codegen_kernel_call(cx, scan_arb, num_threads, scan_arb_args, SHMEM_SIZE, true)

  local postscan_full_args = terralib.newlist()
  postscan_full_args:insertall({lhs_ptr, offset, num_elmts, dir})
  local call_postscan_full =
    gpuhelper.codegen_kernel_call(cx, post_full, num_threads, postscan_full_args, 0, true)

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

-- #####################################
-- ## CUDA Codegen Context
-- #################

local context = {}

function context:__index(field)
  local value = context[field]
  if value ~= nil then
    return value
  end
  error("context has no field '" .. field .. "' (in lookup)", 2)
end

function context:__newindex(field, value)
  error("context has no field '" .. field .. "' (in assignment)", 2)
end

function context.new(use_2d_launch, offset_2d)
  local offset_2d = offset_2d or false
  return setmetatable({
    use_2d_launch = use_2d_launch,
    offset_2d = offset_2d,
    buffered_reductions = data.newmap(),
  }, context)
end

function context:reduction_buffer(ref_type, value_type, op, generator)
  local tbl = self.buffered_reductions[ref_type]
  if tbl == nil then
    tbl = {
      buffer = impl.sharedmemory(value_type, THREAD_BLOCK_SIZE),
      type = value_type,
      op = op,
      generator = generator,
    }
    self.buffered_reductions[ref_type] = tbl
  end
  return tbl
end

function context:compute_reduction_buffer_size()
  local size = 0
  for k, tbl in self.buffered_reductions:items() do
    size = size + sizeof(tbl.type) * THREAD_BLOCK_SIZE
  end
  return size
end

function context:generate_preamble()
  local preamble = terralib.newlist()

  if self.use_2d_launch then
    preamble:insert(quote
      var [self.offset_2d:getsymbol()] = tid_y()
    end)
  end

  local need_barrier = false
  for k, tbl in self.buffered_reductions:items() do
    if tbl.type:isarray() then
      local init = base.reduction_op_init[tbl.op][tbl.type.type]
      preamble:insert(quote
        for k = 0, [tbl.type.N] do
          [tbl.buffer][ tid_y() + tid_x() * [NUM_THREAD_Y] ][k] = [init]
        end
      end)
      need_barrier = true
    else
      local init = base.reduction_op_init[tbl.op][tbl.type]
      preamble:insert(quote
        [tbl.buffer][ tid_y() + tid_x() * [NUM_THREAD_Y] ] = [init]
      end)
      need_barrier = true
    end
  end

  if need_barrier then
    preamble:insert(quote
      impl.barrier()
    end)
  end

  return preamble
end

function context:generate_postamble()
  local postamble = terralib.newlist()

  for k, tbl in self.buffered_reductions:items() do
    postamble:insert(quote
      do
        var tid = tid_y()
        var buf = &[tbl.buffer][ tid_x() * [NUM_THREAD_Y] ]
        barrier()
        [gpuhelper.generate_reduction_tree(tid, buf, NUM_THREAD_Y, tbl.op, tbl.type)]
        if tid == 0 then [tbl.generator(`(@buf))] end
      end
    end)
  end

  return postamble
end

local function check_2d_launch_profitable(node)
  if not base.config["cuda-2d-launch"] or not node:is(ast.typed.stat.ForList) then
    return false, false
  end
  -- TODO: This is a very simple heurstic that does not even extend to 3D case.
  --       At least we need to check if the inner loop has any centered accesses with
  --       respect to that loop. In the longer term, we need a better algorithm to detect
  --       cases where multi-dimensional kernel launches are profitable.
  if #node.block.stats == 1 and node.block.stats[1]:is(ast.typed.stat.ForNum) then
    local inner_loop = node.block.stats[1]
    if inner_loop.metadata and inner_loop.metadata.parallelizable then
      assert(#inner_loop.values == 2)
      return true, base.newsymbol(inner_loop.symbol:gettype(), "offset")
    end
  end
  return false, false
end

function gpuhelper.new_kernel_context(node)
  local use_2d_launch, offset_2d = check_2d_launch_profitable(node)
  return context.new(use_2d_launch, offset_2d)
end

function gpuhelper.optimize_loop(cx, node, block)
  if cx.use_2d_launch then
    local inner_loop = block.stats[1]
    local index_type = inner_loop.symbol:gettype()
    -- If the inner loop is eligible to a 2D kernel launch, we change the stride of the inner
    -- loop accordingly.
    inner_loop = inner_loop {
      values = terralib.newlist({
        ast.typed.expr.Binary {
          op = "+",
          lhs = inner_loop.values[1],
          rhs = ast.typed.expr.ID {
            value = cx.offset_2d,
            expr_type = index_type,
            annotations = ast.default_annotations(),
            span = inner_loop.span,
          },
          expr_type = inner_loop.values[1].expr_type,
          annotations = ast.default_annotations(),
          span = inner_loop.span,
        },
        inner_loop.values[2],
        ast.typed.expr.Constant {
          value = NUM_THREAD_Y,
          expr_type = index_type,
          annotations = ast.default_annotations(),
          span = inner_loop.span,
        }
      })
    }
    block = block { stats = terralib.newlist({ inner_loop }) }
  end
  return block
end

function gpuhelper.generate_region_reduction(cx, loop_symbol, node, rhs, lhs_type, value_type, gen)
  if cx.use_2d_launch then
    local needs_buffer = base.types.is_ref(lhs_type) and
                         (value_type:isprimitive() or value_type:isarray()) and
                         node.metadata and
                         node.metadata.centers and
                         node.metadata.centers:has(loop_symbol)
    if needs_buffer then
      local buffer = cx:reduction_buffer(lhs_type, value_type, node.op, gen).buffer
      return quote
        do
          var idx = tid_y() + tid_x() * [NUM_THREAD_Y]
          [generate_element_reductions(`([buffer][ [idx] ]), rhs, node.op, value_type, false)]
        end
      end
    else
      return gen(rhs)
    end
  else
    return gen(rhs)
  end
end

-- #####################################
-- ## Code generation for kernel argument spill
-- #################

function gpuhelper.check_arguments_need_spill(args)
  local param_size = 0
  args:map(function(arg) param_size = param_size + terralib.sizeof(arg.type) end)
  return param_size > MAX_SIZE_INLINE_KERNEL_PARAMS
end

function gpuhelper.generate_argument_spill(args)
  local arg_type = terralib.types.newstruct("cuda_kernel_arg")
  arg_type.entries = terralib.newlist()
  local mapping = {}
  for i, symbol in pairs(args) do
    local field_name
    field_name = "_arg" .. tostring(i)
    arg_type.entries:insert({ field_name, symbol.type })
    mapping[field_name] = symbol
  end

  local kernel_arg = terralib.newsymbol(&arg_type)
  local buffer = terralib.newsymbol(c.legion_deferred_buffer_char_1d_t, "__spill_buf")
  local buffer_size = sizeof(arg_type)
  buffer_size = (buffer_size + 7) / 8 * 8

  local param_pack = terralib.newlist()
  local param_unpack = terralib.newlist()

  param_pack:insert(quote
    var [kernel_arg]
    var [buffer]
    do
      var bounds : c.legion_rect_1d_t
      bounds.lo.x[0] = 0
      bounds.hi.x[0] = [buffer_size - 1]
      [buffer] = c.legion_deferred_buffer_char_1d_create(bounds, c.Z_COPY_MEM, [&int8](nil))
      [kernel_arg] =
        [&arg_type]([&opaque](c.legion_deferred_buffer_char_1d_ptr([buffer], bounds.lo)))
    end
  end)
  arg_type.entries:map(function(pair)
    local field_name, field_type = unpack(pair)
    local arg = mapping[field_name]
    param_pack:insert(quote (@[kernel_arg]).[field_name] = [arg] end)
    param_unpack:insert(quote var [arg] = (@[kernel_arg]).[field_name] end)
  end)

  local spill_cleanup = quote
    c.legion_deferred_buffer_char_1d_destroy([buffer])
  end

  return param_pack, param_unpack, spill_cleanup, kernel_arg
end

return gpuhelper
