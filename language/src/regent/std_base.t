-- Copyright 2022 Stanford University, NVIDIA Corporation
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

-- Regent Standard Library - Base Layer

local config = require("regent/config")
local data = require("common/data")

local base = {}

base.config, base.args = config.args()

local cpu_fast, gpu_fast = false, "contract"
if terralib.llvm_version < 50 then
  gpu_fast = false
end
if base.config["fast-math"] == 0 then
  cpu_fast, gpu_fast = false, false
elseif base.config["fast-math"] >= 1 then
  cpu_fast, gpu_fast = true, true
end
base.opt_profile = {fastmath = cpu_fast}
base.gpu_opt_profile = {fastmath = gpu_fast}

-- Helpers for zero/min/max values of various types.

local function zero(value_type) return terralib.cast(value_type, 0) end
local function one(value_type) return terralib.cast(value_type, 1) end
local function min_value(value_type)
  if type(rawget(value_type, "min")) == "function" then
    return value_type:min()
  else
    return terralib.cast(value_type, -math.huge)
  end
end
local function max_value(value_type)
  if type(rawget(value_type, "max")) == "function" then
    return value_type:max()
  else
    return terralib.cast(value_type, math.huge)
  end
end

-- #####################################
-- ## Legion Bindings
-- #################

local dlfcn
local function dlopen_library(library_name)
  local ffi = require("ffi")
  if not dlfcn then
    dlfcn = terralib.includec("dlfcn.h")
  end

  -- Right now we do this globally and do not attempt to unload
  -- libraries (and really, there is no safe way to do so because
  -- LuaJIT and LLVM will both get unloaded before we're ready)
  local ok = dlfcn.dlopen(library_name, bit.bor(dlfcn.RTLD_LAZY, dlfcn.RTLD_GLOBAL))
  if ffi.cast("intptr_t", ok) == 0LL then
    assert(false, "dlopen failed: " .. tostring(dlfcn.dlerror()))
  end
end

local function link_library(library_name)
  local suffix = string.sub(library_name, "-3")
  if suffix == ".ll" or suffix == ".bc" then
    terralib.linklibrary(library_name)
  else
    dlopen_library(library_name)
  end
end

do
local linked_libraries = terralib.newlist()

function base.linklibrary(library_name)
  if base.config["offline"] then
    linked_libraries:insert(library_name)
  else
    link_library(library_name)
  end
end

function base.load_all_libraries()
  assert(data.is_luajit())
  linked_libraries:map(function(library)
    link_library(library)
  end)
end

if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  base.binding_library = "libregent.dylib"
else
  base.binding_library = "libregent.so"
end
base.linklibrary(base.binding_library)

end

local c = terralib.includecstring([[
#include "legion.h"
#include "regent.h"
#include "regent_partitions.h"
#include "murmur_hash3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
]])
base.c = c

-- Hack: Terra's parser isn't smart enough to read the value of
-- AUTO_GENERATE_ID, so just force it here so we don't have to
-- hard-code its value elsewhere in the compiler.
c.AUTO_GENERATE_ID = -1

local max_dim = c.LEGION_MAX_DIM
base.max_dim = max_dim

-- #####################################
-- ## Utilities
-- #################

terra base.assert_error(x : bool, message : rawstring)
  if not x then
    var stderr = c.fdopen(2, "w")
    c.fprintf(stderr, "Errors reported during runtime.\n%s\n", message)
    -- Just because it's stderr doesn't mean it's unbuffered...
    c.fflush(stderr)
    c.abort()
  end
end

terra base.assert(x : bool, message : rawstring)
  if not x then
    var stderr = c.fdopen(2, "w")
    c.fprintf(stderr, "assertion failed: %s\n", message)
    -- Just because it's stderr doesn't mean it's unbuffered...
    c.fflush(stderr)
    c.abort()
  end
end

for dim = 1, max_dim do
  local point_type = c["legion_point_" .. tostring(dim) .. "d_t"]
  local rect_type = c["legion_rect_" .. tostring(dim) .. "d_t"]
  local domain_from_rect = c["legion_domain_from_rect_" .. tostring(dim) .. "d"]
  local domain_from_bounds = "domain_from_bounds_" .. tostring(dim) .. "d"

  base[domain_from_bounds] = terra(start : point_type, extent : point_type)
    var rect = rect_type {
      lo = start,
      hi = point_type {
        x = array([data.range(0, dim):map(function(i) return `(start.x[i] + extent.x[i] - 1) end)]),
      },
    }
    return domain_from_rect(rect)
  end
end

-- A whitelist of functions that are known to be ok. We're importing a
-- fixed list of known C headers above, so it should be ok to use a
-- blacklist here.
base.replicable_whitelist = {}
do
  local blacklist = data.set {
    "fprintf",
    "fscanf",
    "printf",
    "scanf",
    "rand",
    "rand_r",
    "vfscanf",
    "vscanf",
  }
  for k, v in pairs(c) do
    if not blacklist[k] then
      base.replicable_whitelist[v] = true
    end
  end

  local other = {
    base.assert_error,
    base.assert,
    base.domain_from_bounds_1d,
    base.domain_from_bounds_2d,
    base.domain_from_bounds_3d,

    -- Terra functions that happen to be placed in global scope:
    _G["sizeof"],
    _G["vector"],
    _G["vectorof"],
    _G["array"],
    _G["arrayof"],
  }
  for _, v in ipairs(other) do
    base.replicable_whitelist[v] = true
  end
end

-- #####################################
-- ## Codegen Helpers
-- #################

function base.normalize_name(name)
  return string.gsub(
    string.gsub(
      string.gsub(name, ".*/", ""),
      "[<>]", ""),
    "[^A-Za-z0-9]", "_")
end

function base.type_meet(a, b)
  local function test()
    local terra query(x : a, y : b)
      if true then return x end
      if true then return y end
    end
    return query:gettype().returntype
  end
  local valid, result_type = pcall(test)

  if valid then
    return result_type
  end
end

local gen_optimal = terralib.memoize(
  function(op, lhs_type, rhs_type)
    return terra(lhs : lhs_type, rhs : rhs_type)
      if [base.quote_binary_op(op, lhs, rhs)] then
        return lhs
      else
        return rhs
      end
    end
  end)

local gen_optimal_skip_nan = terralib.memoize(
  function(op, lhs_type, rhs_type)
    return terra(lhs : lhs_type, rhs : rhs_type)
      -- Do NOT propogate a NaN, if any.
      if [base.quote_binary_op(op, lhs, rhs)] or rhs ~= rhs then
        return lhs
      else
        return rhs
      end
    end
  end)

base.fmax = macro(
  function(lhs, rhs)
    local lhs_type, rhs_type = lhs:gettype(), rhs:gettype()
    local result_type = base.type_meet(lhs_type, rhs_type)
    assert(result_type)
    if result_type:isfloat() then
      return `([gen_optimal_skip_nan(">", lhs_type, rhs_type)]([lhs], [rhs]))
    else
      return `([gen_optimal(">", lhs_type, rhs_type)]([lhs], [rhs]))
    end
  end)

base.fmin = macro(
  function(lhs, rhs)
    local lhs_type, rhs_type = lhs:gettype(), rhs:gettype()
    local result_type = base.type_meet(lhs_type, rhs_type)
    assert(result_type)
    if result_type:isfloat() then
      return `([gen_optimal_skip_nan("<", lhs_type, rhs_type)]([lhs], [rhs]))
    else
      return `([gen_optimal("<", lhs_type, rhs_type)]([lhs], [rhs]))
    end
  end)

function base.quote_unary_op(op, rhs)
  if op == "-" then
    return `(-[rhs])
  elseif op == "not" then
    return `(not [rhs])
  else
    assert(false, "unknown operator " .. tostring(op))
  end
end

function base.quote_binary_op(op, lhs, rhs)
  if op == "*" then
    return `([lhs] * [rhs])
  elseif op == "/" then
    return `([lhs] / [rhs])
  elseif op == "%" then
    return `([lhs] % [rhs])
  elseif op == "+" then
    return `([lhs] + [rhs])
  elseif op == "-" then
    return `([lhs] - [rhs])
  elseif op == "<" then
    return `([lhs] < [rhs])
  elseif op == ">" then
    return `([lhs] > [rhs])
  elseif op == "^" then
    return `([lhs] ^ [rhs])
  elseif op == "<<" then
    return `([lhs] << [rhs])
  elseif op == ">>" then
    return `([lhs] >> [rhs])
  elseif op == "<=" then
    return `([lhs] <= [rhs])
  elseif op == ">=" then
    return `([lhs] >= [rhs])
  elseif op == "==" then
    return `([lhs] == [rhs])
  elseif op == "~=" then
    return `([lhs] ~= [rhs])
  elseif op == "and" then
    return `([lhs] and [rhs])
  elseif op == "or" then
    return `([lhs] or [rhs])
  elseif op == "max" then
    return `([base.fmax]([lhs], [rhs]))
  elseif op == "min" then
    return `([base.fmin]([lhs], [rhs]))
  else
    assert(false, "unknown operator " .. tostring(op))
  end
end

-- #####################################
-- ## Complex types
-- #################

do
  local st = terralib.types.newstruct("complex32")
  st.entries = terralib.newlist({
      { "real", float },
      { "imag", float },
  })
  st.base_type = float
  st.is_complex_type = true
  base.complex = st
  base.complex32 = st
end

do
  local st = terralib.types.newstruct("complex64")
  st.entries = terralib.newlist({
      { "real", double },
      { "imag", double },
  })
  st.base_type = double
  st.is_complex_type = true
  base.complex = st
  base.complex64 = st
end

-- #####################################
-- ## Physical Privilege Helpers
-- #################

-- Physical privileges describe the privileges used by the actual
-- Legion runtime, rather than the privileges used by Regent (as
-- above). Some important differences from normal privileges:
--
--  * Physical privileges are strings (at least for the moment)
--  * Unlike normal privileges, physical privileges form a lattice
--    (with a corresponding meet operator)
--  * "reads_writes" is a physical privilege (not a normal privilege),
--    and is the top of the physical privilege lattice

local function lift(fn)
  return function(value_type)
    if value_type:isarray() then
      return `(array([data.range(value_type.N):map(function(_)
        return lift(fn)(value_type.type)
      end)]))
    elseif value_type == base.complex32 or value_type == base.complex64 then
      return `([value_type] { [fn(value_type.base_type)], 0.0 })
    else
      return fn(value_type)
    end
  end
end

base.reduction_ops = data.map_from_table({
    ["+"] =   { name = "plus",   init = lift(zero)      },
    ["-"] =   { name = "minus",  init = lift(zero)      },
    ["*"] =   { name = "times",  init = lift(one)       },
    ["/"] =   { name = "divide", init = lift(one)       },
    ["max"] = { name = "max",    init = lift(min_value) },
    ["min"] = { name = "min",    init = lift(max_value) },
})
base.reduction_op_ids = {}
base.reduction_op_init = {}
base.all_reduction_ops = terralib.newlist()
base.registered_reduction_ops = terralib.newlist()
do
  local base_op_id = 101
  function base.update_reduction_op(op, op_type, op_id, init)
    if base.reduction_op_ids[op] ~= nil and
       base.reduction_op_ids[op][op_type] ~= nil
    then
      return
    end
    local builtin = op_id ~= nil
    local op_id = op_id or base_op_id
    if not builtin then base_op_id = base_op_id + 1 end
    if not base.reduction_op_ids[op] then
      base.reduction_op_ids[op] = {}
    end
    if not base.reduction_op_init[op] then
      base.reduction_op_init[op] = {}
    end
    base.reduction_op_ids[op][op_type] = op_id
    base.reduction_op_init[op][op_type] = init or base.reduction_ops[op].init(op_type)
    base.all_reduction_ops:insert({op, op_type})
    if not builtin then
      base.registered_reduction_ops:insert({op, op_type})
    end
  end

  -- Prefill the table of reduction op IDs for primitive types.
  local reduction_ops =
    terralib.newlist({ "+", "-", "*", "/", "max", "min" })
  local legion_op_names = data.map_from_table({
    ["+"] = "SUM", ["-"] = "SUM",  ["*"] = "PROD",
    ["/"] = "PROD", ["max"] = "MAX", ["min"] = "MIN" })
  local primitive_reduction_types =
    terralib.newlist({ float, double, int16, int32, int64, uint16, uint32, uint64 })
  for _, op in ipairs(reduction_ops) do
    for _, op_type in ipairs(primitive_reduction_types) do
      local type_name =
        (op_type:isfloat() and ("FLOAT" .. tostring(sizeof(op_type) * 8))) or
        string.upper(tostring(op_type))
      local legion_op_id =
        c["LEGION_REDOP_" .. legion_op_names[op] .. "_" .. type_name]
      base.update_reduction_op(op, op_type, legion_op_id)
    end
  end
  -- Prefill the table of reduction op IDs for complex types.
  do
    base.update_reduction_op("+", base.complex32, c.LEGION_REDOP_SUM_COMPLEX64)
    base.update_reduction_op("-", base.complex32, c.LEGION_REDOP_SUM_COMPLEX64)
    base.update_reduction_op("*", base.complex32, c.LEGION_REDOP_PROD_COMPLEX64)
    base.update_reduction_op("/", base.complex32, c.LEGION_REDOP_PROD_COMPLEX64)

    base.update_reduction_op("+", base.complex64, c.LEGION_REDOP_SUM_COMPLEX128)
    base.update_reduction_op("-", base.complex64, c.LEGION_REDOP_SUM_COMPLEX128)
  end
end

function base.is_reduction_op(privilege)
  assert(type(privilege) == "string")
  return string.sub(privilege, 1, string.len("reduces ")) == "reduces "
end

function base.get_reduction_op(privilege)
  assert(type(privilege) == "string")
  return string.sub(privilege, string.len("reduces ") + 1)
end

-- Assign the basic types IDs for interop with Pygion.
do
  local primitive_types =
    terralib.newlist({ int8, int16, int32, int64, uint8, uint16, uint32, uint64, float, double })
  local base_id = 101
  local type_ids = data.newmap()
  for _, t in ipairs(primitive_types) do
    local type_id = base_id
    base_id = base_id + 1
    type_ids[t] = type_id
  end
  function base.get_type_semantic_tag()
    return 54321 -- Hack: pick a value that seems unlikely to conflict
  end
  function base.get_type_id(t)
    return type_ids[t]
  end
end

function base.meet_privilege(a, b)
  if a == b then
    return a
  elseif not a then
    return b
  elseif not b then
    return a
  elseif a == "none" then
    return b
  elseif b == "none" then
    return a
  else
    return "reads_writes"
  end
end

function base.meet_coherence(a, b)
  if a == b then
    return a
  elseif not a then
    return b
  elseif not b then
    return a
  else
    assert(false)
  end
end

function base.meet_flag(a, b)
  if a == b then
    return a
  elseif not a or a == "no_flag" then
    return b
  elseif not b or b == "no_flag" then
    return a
  else
    assert(false)
  end
end

local function find_field_privilege(privileges, coherence_modes, flags,
                                    region_type, field_path, field_type)
  local field_privilege = "none"
  for _, privilege_list in ipairs(privileges) do
    for _, privilege in ipairs(privilege_list) do
      assert(base.is_symbol(privilege.region))
      assert(data.is_tuple(privilege.field_path))
      if region_type == privilege.region:gettype() and
        field_path:starts_with(privilege.field_path)
      then
        field_privilege = base.meet_privilege(field_privilege,
                                              tostring(privilege.privilege))
      end
    end
  end

  local coherence_mode = "exclusive"
  if coherence_modes[region_type] then
    for prefix, coherence in coherence_modes[region_type]:items() do
      if field_path:starts_with(prefix) then
        coherence_mode = tostring(coherence)
      end
    end
  end

  local flag = "no_flag"
  if flags[region_type] then
    for prefix, flag_fields in flags[region_type]:items() do
      if field_path:starts_with(prefix) then
        for _, flag_kind in flag_fields:keys() do
          flag = base.meet_flag(flag, tostring(flag_kind))
        end
      end
    end
  end

  -- FIXME: Fow now, render write privileges as
  -- read-write. Otherwise, write would get rendered as
  -- write-discard, which would not be correct without explicit
  -- user annotation.
  if field_privilege == "writes" then
    field_privilege = "reads_writes"
  end

  local redop_id = false
  if base.is_reduction_op(field_privilege) then
    local op = base.get_reduction_op(field_privilege)
    if not (base.reduction_op_ids[op] and base.reduction_op_ids[op][field_type]) then
      -- You could upgrade to reads_writes here, but this would never
      -- have made it past the parser anyway.
      assert(false)
    end
    -- Hack: This is a way to bucket reductions by redop ID and make
    -- sure they sort deterministically.
    redop_id = base.reduction_op_ids[op][field_type]
  end

  return field_privilege, redop_id, coherence_mode, flag
end

function base.find_task_privileges(region_type, task)
  assert(base.types.type_supports_privileges(region_type))
  assert(base.is_task(task))

  local privileges = task:get_privileges()
  local coherence_modes = task:get_coherence_modes()
  local flags = task:get_flags()

  local grouped_privileges = terralib.newlist()
  local grouped_coherence_modes = terralib.newlist()
  local grouped_flags = terralib.newlist()
  local grouped_field_paths = terralib.newlist()
  local grouped_field_types = terralib.newlist()

  local field_paths, field_types = base.types.flatten_struct_fields(
    region_type:fspace())

  local fields_by_mode = data.newmap()
  local min_field_index_by_mode = data.newmap()
  local field_type_by_field = data.newmap()
  for i, field_path in ipairs(field_paths) do
    local field_type = field_types[i]
    local privilege, redop_id, coherence, flag = find_field_privilege(
      privileges, coherence_modes, flags, region_type, field_path, field_type)
    local mode = data.newtuple(privilege, redop_id, coherence, flag)
    if privilege ~= "none" then
      if not fields_by_mode[mode] then
        fields_by_mode[mode] = terralib.newlist()
      end
      fields_by_mode[mode]:insert(field_path)
      if not min_field_index_by_mode[mode] then
        min_field_index_by_mode[mode] = i
      end
      field_type_by_field[field_path] = field_type
    end
  end

  -- Sort by mode so we get a stable ordering of the privilege groups.
  local privilege_order = {reads = 1, reads_writes = 2}
  local coherence_order = {exclusive = 1, atomic = 2, simultaneous = 3, relaxed = 4}

  local modes = terralib.newlist()
  for _, mode in fields_by_mode:keys() do
    modes:insert(mode)
  end
  modes:sort(
    function(x, y)
      local px = privilege_order[x[1]]
      local py = privilege_order[y[1]]
      if px and py then
        return px < py
      elseif px then
        return true
      elseif py then
        return false
      elseif x[2] and y[2] and x[2] ~= y[2] then
        -- Reductions are relatively ordered by the field index (i.e.,
        -- the index of the first field that appears in the field
        -- space of the region).
        return min_field_index_by_mode[x] < min_field_index_by_mode[y]
      elseif x[1] ~= y[1] then
        -- There are no other privileges.
        assert(false)
      end

      local cx = coherence_order[x[3]]
      local cy = coherence_order[y[3]]
      if cx and cy then
        return cx < cy
      elseif cx then
        return true
      elseif cy then
        return false
      elseif x[3] ~= y[3] then
        return x[3] < y[3]
      end

      return x[4] < y[4]
    end)

  local privilege_index = data.newmap()
  local privilege_next_index = 1
  for _, mode in ipairs(modes) do
    local privilege, redop_id, coherence, flag = unpack(mode)
    for _, field_path in ipairs(fields_by_mode[mode]) do
      local field_type = field_type_by_field[field_path]
      local index = privilege_index[mode]
      if not index then
        index = privilege_next_index
        privilege_next_index = privilege_next_index + 1

        privilege_index[mode] = index

        grouped_privileges:insert(privilege)
        grouped_coherence_modes:insert(coherence)
        grouped_flags:insert(flag)
        grouped_field_paths:insert(terralib.newlist())
        grouped_field_types:insert(terralib.newlist())
      end

      grouped_field_paths[index]:insert(field_path)
      grouped_field_types[index]:insert(field_type)
    end
  end

  if #grouped_privileges == 0 then
    grouped_privileges:insert("none")
    grouped_coherence_modes:insert("exclusive")
    grouped_flags:insert("no_flag")
    grouped_field_paths:insert(terralib.newlist())
    grouped_field_types:insert(terralib.newlist())
  end

  return grouped_privileges, grouped_field_paths, grouped_field_types,
    grouped_coherence_modes, grouped_flags
end

function base.group_task_privileges_by_field_path(privileges, privilege_field_paths,
                                                  privilege_field_types,
                                                  privilege_coherence_modes,
                                                  privilege_flags)
  local privileges_by_field_path = data.newmap()
  local coherence_modes_by_field_path
  if privilege_coherence_modes ~= nil then
    coherence_modes_by_field_path = data.newmap()
  end
  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    for _, field_path in ipairs(field_paths) do
      privileges_by_field_path[field_path] = privilege
      if coherence_modes_by_field_path ~= nil then
        coherence_modes_by_field_path[field_path] =
          privilege_coherence_modes[i]
      end
    end
  end
  return privileges_by_field_path, coherence_modes_by_field_path
end

local privilege_modes = {
  none            = c.NO_ACCESS,
  reads           = c.READ_ONLY,
  writes          = c.WRITE_ONLY,
  reads_writes    = c.READ_WRITE,
}

function base.privilege_mode(privilege)
  local mode = privilege_modes[privilege]
  if base.is_reduction_op(privilege) then
    mode = c.REDUCE
  end
  assert(mode)
  return mode
end

local coherence_modes = {
  exclusive       = c.EXCLUSIVE,
  atomic          = c.ATOMIC,
  simultaneous    = c.SIMULTANEOUS,
  relaxed         = c.RELAXED,
}

function base.coherence_mode(coherence)
  local mode = coherence_modes[coherence]
  assert(mode)
  return mode
end

local flag_modes = {
  no_flag         = c.NO_FLAG,
  verified_flag   = c.VERIFIED_FLAG,
  no_access_flag  = c.NO_ACCESS_FLAG,
}

function base.flag_mode(flag)
  local mode = flag_modes[flag]
  assert(mode)
  return mode
end

-- #####################################
-- ## Type Helpers
-- #################

base.types = {}

function base.types.is_bounded_type(t)
  return terralib.types.istype(t) and rawget(t, "is_bounded_type") or false
end

function base.types.is_index_type(t)
  return terralib.types.istype(t) and rawget(t, "is_index_type") or false
end

function base.types.is_bounded_type(t)
  return terralib.types.istype(t) and rawget(t, "is_bounded_type") or false
end

function base.types.is_index_type(t)
  return terralib.types.istype(t) and rawget(t, "is_index_type") or false
end

function base.types.is_rect_type(t)
  return terralib.types.istype(t) and rawget(t, "is_rect_type") or false
end

function base.types.is_transform_type(t)
  return terralib.types.istype(t) and rawget(t, "is_transform_type") or false
end

function base.types.is_ispace(t)
  return terralib.types.istype(t) and rawget(t, "is_ispace") or false
end

function base.types.is_region(t)
  return terralib.types.istype(t) and rawget(t, "is_region") or false
end

function base.types.is_partition(t)
  return terralib.types.istype(t) and rawget(t, "is_partition") or false
end

function base.types.is_cross_product(t)
  return terralib.types.istype(t) and rawget(t, "is_cross_product") or false
end

function base.types.is_vptr(t)
  return terralib.types.istype(t) and rawget(t, "is_vpointer") or false
end

function base.types.is_sov(t)
  return terralib.types.istype(t) and rawget(t, "is_struct_of_vectors") or false
end

function base.types.is_ref(t)
  return terralib.types.istype(t) and rawget(t, "is_ref") or false
end

function base.types.is_rawref(t)
  return terralib.types.istype(t) and rawget(t, "is_rawref") or false
end

function base.types.is_future(t)
  return terralib.types.istype(t) and rawget(t, "is_future") or false
end

function base.types.is_list(t)
  return terralib.types.istype(t) and rawget(t, "is_list") or false
end

function base.types.is_list_of_regions(t)
  return base.types.is_list(t) and t:is_list_of_regions()
end

function base.types.is_list_of_partitions(t)
  return base.types.is_list(t) and t:is_list_of_partitions()
end

function base.types.is_list_of_phase_barriers(t)
  return base.types.is_list(t) and t:is_list_of_phase_barriers()
end

function base.types.is_phase_barrier(t)
  return terralib.types.istype(t) and rawget(t, "is_phase_barrier") or false
end

function base.types.is_dynamic_collective(t)
  return terralib.types.istype(t) and rawget(t, "is_dynamic_collective") or false
end

function base.types.is_regent_array(t)
  return terralib.types.istype(t) and rawget(t, "is_regent_array") or false
end

function base.types.is_unpack_result(t)
  return terralib.types.istype(t) and rawget(t, "is_unpack_result") or false
end

function base.types.is_string(t)
  return terralib.types.istype(t) and rawget(t, "is_string") or false
end

function base.types.type_supports_privileges(t)
  return base.types.is_region(t) or base.types.is_list_of_regions(t)
end

function base.types.type_supports_constraints(t)
  return base.types.is_region(t) or base.types.is_partition(t) or
    base.types.is_list_of_regions(t) or base.types.is_list_of_partitions(t)
end

function base.types.type_is_opaque_to_field_accesses(t)
  return base.types.is_region(t) or base.types.is_partition(t) or
    base.types.is_cross_product(t) or base.types.is_list(t)
end

function base.types.is_ctor(t)
  return terralib.types.istype(t) and rawget(t, "is_ctor") or false
end

function base.types.is_fspace_instance(t)
  return terralib.types.istype(t) and rawget(t, "is_fspace_instance") or false
end

function base.types.is_complex_type(t)
  return terralib.types.istype(t) and rawget(t, "is_complex_type") or false
end

function base.types.flatten_struct_fields(struct_type)
  assert(terralib.types.istype(struct_type))
  local field_paths = terralib.newlist()
  local field_types = terralib.newlist()

  local function is_geometric_type(ty)
    return base.types.is_index_type(ty) or base.types.is_rect_type(ty)
  end

  if (struct_type:isstruct() or base.types.is_fspace_instance(struct_type)) and
     not (is_geometric_type(struct_type) or
          base.types.is_regent_array(struct_type) or
          struct_type.__no_field_slicing) then
    local entries = struct_type:getentries()
    for _, entry in ipairs(entries) do
      local entry_name = entry[1] or entry.field
      -- FIXME: Fix for struct types with symbol fields.
      assert(type(entry_name) == "string")
      local entry_type = entry[2] or entry.type
      local entry_field_paths, entry_field_types =
        base.types.flatten_struct_fields(entry_type)
      field_paths:insertall(
        entry_field_paths:map(
          function(entry_field_path)
            return data.newtuple(entry_name) .. entry_field_path
          end))
      field_types:insertall(entry_field_types)
    end
  else
    field_paths:insert(data.newtuple())
    field_types:insert(struct_type)
  end

  return field_paths, field_types
end

-- #####################################
-- ## Symbols
-- #################

local symbol = {}
function symbol:__index(field)
  local value = symbol[field]
  if value ~= nil then return value end
  error("symbol has no field '" .. tostring(field) .. "' (in lookup)", 2)
end

function symbol:__newindex(field, value)
  error("symbol has no field '" .. tostring(field) .. "' (in assignment)", 2)
end

do
  local next_id = 1
  function base.newsymbol(symbol_type, symbol_name)
    -- Swap around the arguments to allow either one to be optional.
    if type(symbol_type) == "string" and symbol_name == nil then
      symbol_type, symbol_name = nil, symbol_type
    elseif symbol_type == nil and terralib.types.istype(symbol_name) then
      symbol_type, symbol_name = symbol_name, nil
    end
    assert(symbol_type == nil or terralib.types.istype(symbol_type), "newsymbol expected argument 1 to be a type")
    assert(symbol_name == nil or type(symbol_name) == "string", "newsymbol expected argument 2 to be a string")

    local id = next_id
    next_id = next_id + 1
    return setmetatable({
      symbol_type = symbol_type or false,
      symbol_name = symbol_name or false,
      symbol_symbol = false,
      symbol_label = false,
      symbol_id = id,
    }, symbol)
  end
end

function base.is_symbol(x)
  return getmetatable(x) == symbol
end

function symbol:hasname()
  return self.symbol_name or nil
end

function symbol:getname()
  assert(self.symbol_name)
  return self.symbol_name
end

function symbol:hastype()
  return self.symbol_type or nil
end

function symbol:gettype()
  assert(self.symbol_type)
  return self.symbol_type
end

function symbol:settype(type)
  assert(terralib.types.istype(type))
  assert(not self.symbol_type)
  assert(not self.symbol_symbol)
  self.symbol_type = type
end

function symbol:getsymbol()
  assert(self.symbol_type)
  if not self.symbol_symbol then
    self.symbol_symbol = terralib.newsymbol(self.symbol_type, self.symbol_name)
  end
  return self.symbol_symbol
end

function symbol:getlabel()
  if not self.symbol_label then
    self.symbol_label = terralib.newlabel(self.symbol_name)
  end
  return self.symbol_label
end

function symbol:__tostring()
  if self:hasname() then
    if base.config["debug"] then
      return "$" .. tostring(self:getname()) .. "#" .. tostring(self.symbol_id)
    else
      return "$" .. tostring(self:getname())
    end
  else
    return "$" .. tostring(self.symbol_id)
  end
end

-- #####################################
-- ## Variants
-- #################

base.variant = {}
function base.variant:__index(field)
  local value = base.variant[field]
  if value ~= nil then return value end
  error("variant has no field '" .. field .. "' (in lookup)", 2)
end

function base.variant:__newindex(field, value)
  error("variant has no field '" .. field .. "' (in assignment)", 2)
end

function base.variant:set_variant_id(variant_id)
  self.variant_id = variant_id
end

function base.variant:get_variant_id()
  return self.variant_id
end

function base.variant:set_is_cuda(cuda)
  self.cuda = cuda
end

function base.variant:is_cuda()
  return self.cuda
end

function base.variant:set_is_openmp(openmp)
  self.openmp = openmp
end

function base.variant:is_openmp()
  return self.openmp
end

function base.variant:set_is_inline(inline)
  self.inline = inline
end

function base.variant:is_inline()
  return self.inline
end

do
  -- We use this counter to guarantee that each CUDA kernel within a module has a unique name
  -- Note that task-local counters prefixed by task names are insufficient because meta-programmed
  -- tasks have the same name.
  local global_kernel_counter = 1
  function base.variant:add_cuda_kernel(kernel)
    if not self.cudakernels then
      self.cudakernels = terralib.newlist()
    end
    local kernel_name = self.task:get_name():concat("_") .. "_cuda" .. tostring(global_kernel_counter)
    kernel_name = "__kernel_id_" .. kernel_name
    self.cudakernels:insert({
      name = kernel_name,
      kernel = kernel,
    })
    kernel:setname(kernel_name)
    global_kernel_counter = global_kernel_counter + 1
    return kernel_name
  end
end

function base.variant:get_cuda_kernels()
  return self.cudakernels or terralib.newlist()
end

function base.variant:set_config_options(t)
  assert(not self.config_options)
  self.config_options = t
end

function base.variant:get_config_options()
  self.task:complete()
  assert(self.config_options)
  return self.config_options
end

function base.variant:get_name()
  return self.name
end

function base.variant:get_definition()
  self.task:complete()
  assert(self.definition)
  return self.definition
end

function base.variant:set_definition(definition)
  assert(not self.definition)
  self.definition = definition
end

function base.variant:set_ast(ast)
  assert(not self.ast)
  self.ast = ast
end

function base.variant:has_ast()
  return self.ast
end

function base.variant:get_ast()
  assert(self.ast)
  return self.ast
end

function base.variant:set_untyped_ast(ast)
  assert(not self.untyped_ast)
  self.untyped_ast = ast
end

function base.variant:has_untyped_ast()
  return self.untyped_ast
end

function base.variant:get_untyped_ast()
  assert(self.untyped_ast)
  return self.untyped_ast
end

function base.variant:compile()
  if base.config["offline"] then
    error("Manual compile requests for individual task variants are prohibited in the offline mode.")
  end
  self.task:complete()
  return self:get_definition():compile()
end

function base.variant:disas()
  self.task:complete()
  return self:get_definition():disas()
end

function base.variant:wrapper_name()
  -- Must be an alphanumeric symbol, because it will be communicated through a
  -- (generated) header file.
  return
    '__regent_task'
    .. '_' .. self.task.unique_task_identifier
    .. '_' .. self:get_name()
end

function base.variant:wrapper_sig()
  return 'void ' .. self:wrapper_name() .. '( void* data'
                                        .. ', size_t datalen'
                                        .. ', void* userdata'
                                        .. ', size_t userlen'
                                        .. ', legion_proc_id_t proc_id'
                                        .. ');'
end

-- This is a type representing a buffer containing a serialized value.
-- The value owns the buffer.
struct base.serialized_value {
  value: &opaque,
  size: uint64,
}

local function make_task_wrapper(task_body)
  return terra(data : &opaque, datalen : c.size_t,
               userdata : &opaque, userlen : c.size_t,
               proc_id : c.legion_proc_id_t)
    var task : c.legion_task_t,
        regions : &c.legion_physical_region_t,
        num_regions : uint32,
        ctx : c.legion_context_t,
        runtime : c.legion_runtime_t
    c.legion_task_preamble(data, datalen, proc_id, &task, &regions, &num_regions, &ctx, &runtime)
    var result = base.serialized_value { nil, 0 }
    task_body(task, regions, num_regions, ctx, runtime, &result)
    c.legion_task_postamble(runtime, ctx, result.value, result.size)
    c.free(result.value)
  end
end

-- Generate task wrapper on this process (it will be compiled automatically).
function base.variant:make_wrapper()
  local wrapper = make_task_wrapper(self:get_definition())
  wrapper:setname(self:wrapper_name())
  return wrapper
end

function base.variant:__tostring()
  return tostring(self.task:get_name()) .. '_' .. self:get_name()
end

function base.variant:add_layout_constraint(constraint)
  if not self.layout_constraints then
    self.layout_constraints = terralib.newlist()
  end
  self.layout_constraints:insert(constraint)
end

function base.variant:has_layout_constraints()
  return self.layout_constraints
end

function base.variant:get_layout_constraints()
  assert(self.layout_constraints)
  return self.layout_constraints
end

function base.variant:add_execution_constraint(constraint)
  if not self.execution_constraints then
    self.execution_constraints = terralib.newlist()
  end
  self.execution_constraints:insert(constraint)
end

function base.variant:has_execution_constraints()
  return self.execution_constraints
end

function base.variant:get_execution_constraints()
  assert(self.execution_constraints)
  return self.execution_constraints
end

do
  function base.new_variant(task, name)
    assert(base.is_task(task))
    assert(type(name) == "string" and not name:match("%W"))

    local variant = setmetatable({
      task = task,
      name = name,
      ast = false,
      untyped_ast = false,
      definition = false,
      cuda = false,
      openmp = false,
      inline = false,
      cudakernels = false,
      config_options = false,
      layout_constraints = false,
      execution_constraints = false,
      variant_id = false,
    }, base.variant)

    task.variants:insert(variant)
    return variant
  end
end

function base.is_variant(x)
  return getmetatable(x) == base.variant
end

-- #####################################
-- ## Tasks
-- #################

base.initial_regent_task_id = 10000

local make_unique_task_identifier
do
  local unique_task_ids = {}
  function make_unique_task_identifier(name)
    local id = base.normalize_name(tostring(name))

    if not unique_task_ids[id] then
      unique_task_ids[id] = true
      return id
    end

    local offset = 1
    local id_offset
    repeat
      id_offset = id .. "_" .. offset
      offset = offset + 1
    until not unique_task_ids[id_offset]
    unique_task_ids[id_offset] = true
    return id_offset
  end
end

base.task = {}
function base.task:__index(field)
  local value = base.task[field]
  if value ~= nil then return value end
  error("task has no field '" .. field .. "' (in lookup)", 2)
end

function base.task:__newindex(field, value)
  error("task has no field '" .. field .. "' (in assignment)", 2)
end

function base.task:set_param_symbols(symbols, force)
  assert(force or not self.param_symbols)
  assert(terralib.islist(symbols))
  self.param_symbols = symbols
end

function base.task:get_param_symbols()
  assert(self.param_symbols)
  return self.param_symbols
end

function base.task:set_params_struct(t)
  assert(not self.params_struct)
  assert(terralib.types.istype(t))
  self.params_struct = t
end

function base.task:get_params_struct()
  assert(self.params_struct)
  return self.params_struct
end

function base.task:set_params_map_type(t)
  assert(not self.params_map_type)
  assert(terralib.types.istype(t))
  self.params_map_type = t
end

function base.task:has_params_map_type()
  return self.params_map_type
end

function base.task:get_params_map_type()
  assert(self.params_map_type)
  return self.params_map_type
end

function base.task:set_params_map_label(label)
  assert(not self.params_map_label)
  assert(terralib.islabel(label))
  self.params_map_label = label
end

function base.task:has_params_map_label()
  return self.params_map_label
end

function base.task:get_params_map_label()
  assert(self.params_map_label)
  return self.params_map_label
end

function base.task:set_params_map_symbol(symbol)
  assert(not self.params_map_symbol)
  assert(terralib.issymbol(symbol))
  self.params_map_symbol = symbol
end

function base.task:has_params_map_symbol()
  return self.params_map_symbol
end

function base.task:get_params_map_symbol()
  assert(self.params_map_symbol)
  return self.params_map_symbol
end

function base.task:set_field_id_param_labels(t)
  assert(not self.field_id_param_labels)
  assert(t)
  self.field_id_param_labels = t
end

function base.task:get_field_id_param_labels()
  assert(self.field_id_param_labels)
  return self.field_id_param_labels
end

function base.task:set_type(t, force)
  assert(force or not self.type)
  assert(terralib.types.istype(t))
  self.type = t
end

function base.task:get_type()
  assert(self.type)
  return self.type
end

function base.task:set_privileges(t)
  assert(not self.privileges)
  assert(terralib.islist(t))
  self.privileges = t
end

function base.task:get_privileges()
  assert(self.privileges)
  return self.privileges
end

function base.task:set_coherence_modes(t)
  assert(not self.coherence_modes)
  assert(t)
  self.coherence_modes = t
end

function base.task:get_coherence_modes()
  assert(self.coherence_modes)
  return self.coherence_modes
end

function base.task:set_flags(t)
  assert(not self.flags)
  assert(t)
  self.flags = t
end

function base.task:get_flags()
  assert(self.flags)
  return self.flags
end

function base.task:set_conditions(conditions)
  assert(not self.conditions)
  assert(conditions)
  self.conditions = conditions
end

function base.task:get_conditions()
  assert(self.conditions)
  return self.conditions
end

function base.task:set_param_constraints(t)
  assert(not self.param_constraints)
  self.param_constraints = t
end

function base.task:get_param_constraints()
  assert(self.param_constraints)
  return self.param_constraints
end

function base.task:set_constraints(t)
  assert(not self.constraints)
  assert(t)
  self.constraints = t
end

function base.task:get_constraints()
  assert(self.constraints)
  return self.constraints
end

function base.task:set_region_universe(t)
  assert(not self.region_universe)
  assert(t)
  self.region_universe = t
end

function base.task:get_region_universe()
  assert(self.region_universe)
  return self.region_universe
end

function base.task:set_task_id(task_id)
  assert(not self.is_complete)

  -- This is intended for interop with tasks defined externally. It
  -- would be dangerous to call this on a Regent task with variants,
  -- because the task ID might already be baked into the
  -- implementation of some task.
  if #self:get_variants() > 0 then
    error("task ID can only be set when task has zero variants")
  end
  if base.config["separate"] then
    self.taskid:setinitializer(task_id)
  else
    self.taskid = terralib.constant(c.legion_task_id_t, task_id)
  end
end

-- TODO: This is actually safe once we make task ids global variables
function base.task:set_task_id_unsafe(task_id)
  if base.config["separate"] then
    self.taskid:setinitializer(task_id)
  else
    self.taskid = terralib.constant(c.legion_task_id_t, task_id)
  end
end

function base.task:get_task_id()
  return self.taskid
end

function base.task:has_mapper_id()
  return self.mapper_id
end

function base.task:get_mapper_id()
  assert(self.mapper_id)
  return self.mapper_id
end

function base.task:set_mapper_id(mapper_id)
  if base.config["separate"] then
    self.mapper_id:setinitializer(mapper_id)
  else
    self.mapper_id = terralib.constant(c.legion_mapper_id_t, mapper_id)
  end
end

function base.task:has_mapping_tag_id()
  return self.mapping_tag_id
end

function base.task:get_mapping_tag_id()
  assert(self.mapping_tag_id)
  return self.mapping_tag_id
end

function base.task:set_mapping_tag_id(tag)
  if base.config["separate"] then
    self.mapping_tag_id:setinitializer(tag)
  else
    self.mapping_tag_id = terralib.constant(c.legion_mapping_tag_id_t, tag)
  end
end

function base.task:set_name(name)
  if type(name) == "string" then
    name = data.newtuple(name)
  elseif data.is_tuple(name) then
    assert(data.all(name:map(function(n) return type(n) == "string" end)))
  else
    assert(false)
  end

  self.name = name
  self.unique_task_identifier = make_unique_task_identifier(name)
  if base.config["separate"] then
    self.taskid:setname("__regent_task_" .. self.unique_task_identifier .. "_task_id")
  end
end

function base.task:get_name()
  return self.name
end

function base.task:is_extern()
  return self.extern
end

function base.task:has_calling_convention()
  return self.calling_convention
end

function base.task:get_calling_convention()
  assert(not self.is_complete)
  assert(self.calling_convention)
  return self.calling_convention
end

function base.task:get_variants()
  return self.variants
end

function base.task:get_variant(name)
  local variant = nil
  for i = 1, #self.variants do
    if self.variants[i]:get_name() == name then
      variant = self.variants[i]
      break
    end
  end
  if variant == nil then
    error("variant '" .. name .. "' does not exist")
  end
  return variant
end

function base.task:set_primary_variant(task)
  assert(not self.primary_variant and not self:is_extern())
  self.primary_variant = task
end

function base.task:has_primary_variant()
  return self.primary_variant
end

function base.task:get_primary_variant()
  assert(self.primary_variant)
  return self.primary_variant
end

function base.task:set_cuda_variant(task)
  assert(not self.cuda_variant and not self:is_extern())
  self.cuda_variant = task
end

function base.task:get_cuda_variant()
  return self.cuda_variant
end

function base.task:set_parallel_task(task)
  assert(not self.parallel_task and not self:is_extern())
  self.parallel_task = task
end

function base.task:get_parallel_task()
  return self.parallel_task
end

function base.task:is_shard_task()
  -- FIXME: This will break if we pick different names for shard tasks
  return string.sub(tostring(self:get_name()), 0, 6) == "<shard"
end

function base.task:make_variant(name)
  assert(not self.is_complete)
  local variant = base.new_variant(self, name)
  return variant
end

function base.task:set_compile_thunk(compile_thunk)
  assert(not self.is_complete)
  self.compile_thunk = compile_thunk
end

function base.task:complete()
  if not self.is_inline and not self.is_complete then
    self.is_complete = true
    if not self.compile_thunk then return end
    for _, variant in ipairs(self.variants) do
      self.compile_thunk(variant)
    end
  end
  return self
end

function base.task:compile()
  return self:complete()
end

function base.task:set_is_inline(is_inline)
  self.is_inline = is_inline
end

function base.task:set_is_local(is_local)
  self.is_local = is_local
end

function base.task:set_optimization_thunk(optimization_thunk)
  self.optimization_thunk = optimization_thunk
end

function base.task:optimize()
  if self.is_inline then
    if not self.optimization_thunk then return self end
    self.optimization_thunk()
    self.is_inline = false
  end
  return self
end

function base.task:__tostring()
  return tostring(self:get_name())
end

do
  local next_task_id
  if not base.config["separate"] then
    next_task_id = base.initial_regent_task_id
  end
  function base.new_task(name, span, extern)
    if type(name) == "string" then
      name = data.newtuple(name)
    elseif data.is_tuple(name) then
      assert(data.all(name:map(function(n) return type(n) == "string" end)))
    else
      assert(false)
    end

    local unique_id = make_unique_task_identifier(name)
    local task_id
    if base.config["separate"] then
      local global_name = "__regent_task_" .. unique_id .. "_task_id"
      if extern then
        task_id = terralib.global(c.legion_task_id_t, nil, global_name, true)
      else
        task_id = terralib.global(c.legion_task_id_t, c.AUTO_GENERATE_ID, global_name)
      end
    else
      task_id = terralib.constant(c.legion_task_id_t, next_task_id)
      next_task_id = next_task_id + 1
    end

    local mapper_id = false
    if base.config["separate"] then
      local global_name = "__regent_task_" .. unique_id .. "_mapper_id"
      if extern then
        mapper_id = terralib.global(c.legion_mapper_id_t, nil, global_name, true)
      else
        mapper_id = terralib.global(c.legion_mapper_id_t, 0, global_name)
      end
    end

    local mapping_tag_id = false
    if base.config["separate"] then
      local global_name = "__regent_task_" .. unique_id .. "_mapping_tag_id"
      if extern then
        mapping_tag_id = terralib.global(c.legion_mapping_tag_id_t, nil, global_name, true)
      else
        mapping_tag_id = terralib.global(c.legion_mapping_tag_id_t, 0, global_name)
      end
    end

    return setmetatable({
      name = name,
      span = span,
      extern = extern or false,
      unique_task_identifier = unique_id,
      taskid = task_id,
      variants = terralib.newlist(),
      calling_convention = false,

      -- User-configurable task metadata:
      mapper_id = mapper_id,
      mapping_tag_id = mapping_tag_id,

      -- Metadata for the Regent calling convention:
      param_symbols = false,
      params_struct = false,
      params_map_type = false,
      params_map_label = false,
      params_map_symbol = false,
      field_id_param_labels = false,
      field_id_param_symbols = false,

      -- Task metadata:
      type = false,
      privileges = false,
      coherence_modes = false,
      flags = false,
      conditions = false,
      param_constraints = false,
      constraints = false,
      region_universe = false,

      -- Variants and alternative versions:
      primary_variant = false,
      cuda_variant = false,
      parallel_task = false,

      -- Compilation continuations:
      compile_thunk = false,
      is_complete = false,
      optimization_thunk = false,
      is_inline = false,
      is_local = false,
    }, base.task)
  end
end

function base.is_task(x)
  return getmetatable(x) == base.task
end

return base
