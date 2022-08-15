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

local base = require("regent/std_base")
local std = require("regent/std")

local omp = {}

-- Exit early if the user turned off OpenMP code generation

if std.config["openmp"] == 0 then
  function omp.check_openmp_available()
    return false
  end
  return omp
end

local has_openmp = true
if not (std.config["offline"] or std.config["openmp-offline"]) then
  local dlfcn = terralib.includec("dlfcn.h")
  local terra find_openmp_symbols()
    var lib = dlfcn.dlopen([&int8](0), dlfcn.RTLD_LAZY)
    var has_openmp =
      dlfcn.dlsym(lib, "GOMP_parallel") ~= [&opaque](0) and
      dlfcn.dlsym(lib, "omp_get_num_threads") ~= [&opaque](0) and
      dlfcn.dlsym(lib, "omp_get_max_threads") ~= [&opaque](0) and
      dlfcn.dlsym(lib, "omp_get_thread_num") ~= [&opaque](0)
    dlfcn.dlclose(lib)
    return has_openmp
  end
  has_openmp = find_openmp_symbols()
end

if not has_openmp then
  function omp.check_openmp_available()
    return false, "Regent is installed without OpenMP support"
  end
  if std.config["openmp"] == 1 then
    local available, message = omp.check_openmp_available()
    print("OpenMP code generation failed since " .. message)
    os.exit(-1)
  end

else
  omp.check_openmp_available = function() return true end
  local omp_abi = terralib.includecstring [[
    extern int omp_get_num_threads(void);
    extern int omp_get_max_threads(void);
    extern int omp_get_thread_num(void);
    extern void GOMP_parallel(void (*fnptr)(void *data), void *data, int nthreads, unsigned flags);
  ]]

  omp.get_num_threads = omp_abi.omp_get_num_threads
  omp.get_max_threads = omp_abi.omp_get_max_threads
  omp.get_thread_num = omp_abi.omp_get_thread_num
  omp.launch = omp_abi.GOMP_parallel
end

-- TODO: This might not be the right size in platforms other than x86
omp.CACHE_LINE_SIZE = 64

local FAST_ATOMICS = {
  ["+"] = "add",
  ["-"] = "sub",
}

omp.generate_atomic_update = terralib.memoize(function(op, typ)
  -- Build a C wrapper to use atomic intrinsics in LLVM
  local atomic_update = nil
  local op_name = base.reduction_ops[op].name
  assert(op_name ~= nil)
  -- Integer types
  if typ:isintegral() then
    local ctype = typ.cachedcstring or typ:cstring()
    assert(ctype ~= nil)
    -- If there is a native support for the operation, use it directly
    if FAST_ATOMICS[op] ~= nil then
      local fun_name = string.format("__atomic_update_%s_%s", op_name, ctype)
      local C = terralib.includecstring(string.format([[
        #include <stdint.h>
        inline void %s(%s *address, %s val) {
          __sync_fetch_and_%s(address, val);
        }
      ]], fun_name, ctype, ctype, FAST_ATOMICS[op]))
      terra atomic_update(address : &typ, val : typ)
        [ C[fun_name] ](address, val)
      end
    else
      local fun_name = string.format("__compare_and_swap_%s_%s", op_name, ctype)
      local C = terralib.includecstring(string.format([[
        #include <stdint.h>
        inline %s %s(%s *address, %s old, %s new) {
          return __sync_val_compare_and_swap(address, old, new);
        }
      ]], ctype, fun_name, ctype, ctype, ctype))
      terra atomic_update(address : &typ, val : typ)
        var success = false
        while not success do
          var old = @address
          var new = [std.quote_binary_op(op, old, val)]
          var res = [ C[fun_name] ](address, old, new)
          success = res == old
        end
      end
    end
  else
    local size = terralib.sizeof(typ) * 8
    local cas_type = _G["uint" .. tostring(size)]
    local ctype = typ.cachedcstring or typ:cstring()
    local cas_ctype = cas_type.cachedcstring or cas_type:cstring()
    local fun_name = string.format("__compare_and_swap_%s_%s", op_name, ctype)
    local C = terralib.includecstring(string.format([[
      #include <stdint.h>
      inline %s %s(%s *address, %s old, %s new) {
        return __sync_val_compare_and_swap(address, old, new);
      }
    ]], cas_ctype, fun_name, cas_ctype, cas_ctype, cas_ctype))
    terra atomic_update(address : &typ, val : typ)
      var success = false
      while not success do
        var old = @address
        var new = [std.quote_binary_op(op, old, val)]

        var address_b : &cas_type = [&cas_type](address)
        var old_b : &cas_type = [&cas_type](&old)
        var new_b : &cas_type = [&cas_type](&new)
        var res : cas_type = [ C[fun_name] ](address_b, @old_b, @new_b)
        success = res == @old_b
      end
    end
  end
  assert(atomic_update ~= nil)
  atomic_update:setinlined(true)
  return atomic_update
end)

function omp.generate_preamble(rect, idx, start_idx, end_idx)
  return quote
    var num_threads = [omp.get_num_threads]()
    var thread_id = [omp.get_thread_num]()
    var lo = [rect].lo.x[idx]
    var hi = [rect].hi.x[idx] + 1
    var chunk = (hi - lo + num_threads - 1) / num_threads
    if chunk == 0 then chunk = 1 end
    var [start_idx] = thread_id * chunk + lo
    var [end_idx] = (thread_id + 1) * chunk + lo
    if [end_idx] > hi then [end_idx] = hi end
  end
end

function omp.generate_argument_type(symbols, reductions)
  local arg_type = terralib.types.newstruct("omp_worker_arg")
  arg_type.entries = terralib.newlist()
  local mapping = {}
  for i, symbol in ipairs(symbols) do
    local field_name
    if reductions[symbol] == nil then
      field_name = "_arg" .. tostring(i)
      arg_type.entries:insert({ field_name, symbol.type })
    else
      field_name = "_red" .. tostring(i)
      arg_type.entries:insert({ field_name, &symbol.type })
    end
    mapping[field_name] = symbol
  end
  return arg_type, mapping
end

function omp.generate_argument_init(arg, arg_type, mapping, can_change, reductions)
  local worker_init = arg_type.entries:map(function(pair)
    local field_name, field_type = unpack(pair)
    local symbol = mapping[field_name]
    if reductions[symbol] ~= nil then
      local init = std.reduction_op_init[reductions[symbol]][symbol.type]
      return quote var [symbol] = [init] end
    else
      return quote var [symbol] = [arg].[field_name] end
    end
  end)

  local launch_init = terralib.newlist()
  launch_init:insert(quote
    var arg_obj : arg_type
    var [arg] = &arg_obj
  end)
  local launch_update = terralib.newlist()

  arg_type.entries:map(function(pair)
    local field_name, field_type = unpack(pair)
    local symbol = mapping[field_name]
    if reductions[symbol] ~= nil then
      local init = std.reduction_op_init[reductions[symbol]][symbol.type]
      assert(field_type:ispointer())
      launch_init:insert(quote
        var num_threads = [omp.get_max_threads]()
        -- We don't like false sharing
        var size = num_threads  * omp.CACHE_LINE_SIZE
        var data = std.c.malloc(size)
        std.assert(size == 0 or data ~= nil, "malloc failed in generate_argument_init")
        [arg].[field_name] = [field_type](data)
        for i = 0, num_threads do
          @[&symbol.type]([&int8](data) + i * omp.CACHE_LINE_SIZE) = [init]
        end
      end)
    elseif not can_change[symbol] then
      launch_init:insert(quote [arg].[field_name] = [symbol] end)
    else
      launch_update:insert(quote [arg].[field_name] = [symbol] end)
    end
  end)

  return worker_init, launch_init, launch_update
end

function omp.generate_worker_cleanup(arg, arg_type, mapping, reductions)
  return arg_type.entries:map(function(pair)
    local field_name, field_type = unpack(pair)
    local symbol = mapping[field_name]
    local op = reductions[symbol]
    if op ~= nil then
      return quote
        do
          var idx = [omp.get_thread_num]() * (omp.CACHE_LINE_SIZE / [sizeof(symbol.type)])
          [arg].[field_name][idx] = [std.quote_binary_op(op, symbol,
            `([arg].[field_name][idx]))]
        end
      end
    else
      return quote end
    end
  end)
end

function omp.generate_launcher_cleanup(arg, arg_type, mapping, reductions)
  return arg_type.entries:map(function(pair)
    local field_name, field_type = unpack(pair)
    local symbol = mapping[field_name]
    local op = reductions[symbol]
    if op ~= nil then
      return quote
        for i = 0, [omp.get_max_threads]() do
          var idx = i * (omp.CACHE_LINE_SIZE / [sizeof(symbol.type)])
          [symbol] = [std.quote_binary_op(op, symbol, `([arg].[field_name][idx]))]
        end
        std.c.free([arg].[field_name])
      end
    else
      return quote end
    end
  end)
end

return omp
