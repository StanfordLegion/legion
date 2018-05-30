-- Copyright 2018 Stanford University, NVIDIA Corporation
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


-- Hack: Terra symbols don't support the hash() method so monkey patch
-- it in here. This allows deterministic hashing of Terra symbols,
-- which is currently required by OpenMP codegen.
do
  local terralib_symbol = getmetatable(terralib.newsymbol(int))
  function terralib_symbol:hash()
    local hash_value = "__terralib_symbol_#" .. tostring(self.id)
    return hash_value
  end
end

-- #####################################
-- ## Legion Bindings
-- #################

terralib.linklibrary("libregent.so")
local c = terralib.includecstring([[
#include "legion.h"
#include "legion_terra.h"
#include "legion_terra_partitions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
]])
base.c = c

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

terra base.domain_from_bounds_1d(start : c.legion_point_1d_t,
                                 extent : c.legion_point_1d_t)
  var rect = c.legion_rect_1d_t {
    lo = start,
    hi = c.legion_point_1d_t {
      x = array(start.x[0] + extent.x[0] - 1),
    },
  }
  return c.legion_domain_from_rect_1d(rect)
end

terra base.domain_from_bounds_2d(start : c.legion_point_2d_t,
                                 extent : c.legion_point_2d_t)
  var rect = c.legion_rect_2d_t {
    lo = start,
    hi = c.legion_point_2d_t {
      x = array(start.x[0] + extent.x[0] - 1,
                start.x[1] + extent.x[1] - 1),
    },
  }
  return c.legion_domain_from_rect_2d(rect)
end

terra base.domain_from_bounds_3d(start : c.legion_point_3d_t,
                                 extent : c.legion_point_3d_t)
  var rect = c.legion_rect_3d_t {
    lo = start,
    hi = c.legion_point_3d_t {
      x = array(start.x[0] + extent.x[0] - 1,
                start.x[1] + extent.x[1] - 1,
                start.x[2] + extent.x[2] - 1),
    },
  }
  return c.legion_domain_from_rect_3d(rect)
end

-- #####################################
-- ## Codegen Helpers
-- #################

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

base.fmax = macro(
  function(lhs, rhs)
    local lhs_type, rhs_type = lhs:gettype(), rhs:gettype()
    local result_type = base.type_meet(lhs_type, rhs_type)
    assert(result_type)
    return `([gen_optimal(">", lhs_type, rhs_type)]([lhs], [rhs]))
  end)

base.fmin = macro(
  function(lhs, rhs)
    local lhs_type, rhs_type = lhs:gettype(), rhs:gettype()
    local result_type = base.type_meet(lhs_type, rhs_type)
    assert(result_type)
    return `([gen_optimal("<", lhs_type, rhs_type)]([lhs], [rhs]))
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

local function zero(value_type) return terralib.cast(value_type, 0) end
local function one(value_type) return terralib.cast(value_type, 1) end
local function min_value(value_type) return terralib.cast(value_type, -math.huge) end
local function max_value(value_type) return terralib.cast(value_type, math.huge) end

base.reduction_ops = terralib.newlist({
    {op = "+", name = "plus", init = zero},
    {op = "-", name = "minus", init = zero},
    {op = "*", name = "times", init = one},
    {op = "/", name = "divide", init = one},
    {op = "max", name = "max", init = min_value},
    {op = "min", name = "min", init = max_value},
})

base.reduction_types = terralib.newlist({
    float,
    double,
    int32,
    int64,
    uint32,
    uint64,
})

base.reduction_op_init = {}
for _, op in ipairs(base.reduction_ops) do
  base.reduction_op_init[op.op] = {}
  for _, op_type in ipairs(base.reduction_types) do
    base.reduction_op_init[op.op][op_type] = op.init(op_type)
  end
end

-- Prefill the table of reduction op IDs.
base.reduction_op_ids = {}
do
  local base_op_id = 101
  for _, op in ipairs(base.reduction_ops) do
    for _, op_type in ipairs(base.reduction_types) do
      local op_id = base_op_id
      base_op_id = base_op_id + 1
      if not base.reduction_op_ids[op.op] then
        base.reduction_op_ids[op.op] = {}
      end
      base.reduction_op_ids[op.op][op_type] = op_id
    end
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

  if base.is_reduction_op(field_privilege) then
    local op = base.get_reduction_op(field_privilege)
    if not (base.reduction_op_ids[op] and base.reduction_op_ids[op][field_type]) then
      -- You could upgrade to reads_writes here, but this would never
      -- have made it past the parser anyway.
      assert(false)
    end
  end

  return field_privilege, coherence_mode, flag
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

  local privilege_index = data.newmap()
  local privilege_next_index = 1
  for i, field_path in ipairs(field_paths) do
    local field_type = field_types[i]
    local privilege, coherence, flag = find_field_privilege(
      privileges, coherence_modes, flags, region_type, field_path, field_type)
    local mode = data.newtuple(privilege, coherence, flag)
    if privilege ~= "none" then
      local index = privilege_index[mode]
      if not index then
        index = privilege_next_index
        privilege_next_index = privilege_next_index + 1

        -- Reduction privileges cannot be grouped, because the Legion
        -- runtime does not know how to handle multi-field reductions.
        if not base.is_reduction_op(privilege) then
          privilege_index[mode] = index
        end

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
  local privileges_by_field_path = {}
  local coherence_modes_by_field_path
  if privilege_coherence_modes ~= nil then
    coherence_modes_by_field_path = {}
  end
  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    for _, field_path in ipairs(field_paths) do
      privileges_by_field_path[field_path:hash()] = privilege
      if coherence_modes_by_field_path ~= nil then
        coherence_modes_by_field_path[field_path:hash()] =
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

function base.types.flatten_struct_fields(struct_type)
  assert(terralib.types.istype(struct_type))
  local field_paths = terralib.newlist()
  local field_types = terralib.newlist()

  local function is_geometric_type(ty)
    return base.types.is_index_type(ty) or base.types.is_rect_type(ty)
  end

  if (struct_type:isstruct() or base.types.is_fspace_instance(struct_type)) and
     not is_geometric_type(struct_type) then
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
  error("symbol has no field '" .. field .. "' (in lookup)", 2)
end

function symbol:__newindex(field, value)
  error("symbol has no field '" .. field .. "' (in assignment)", 2)
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

function symbol:hash()
  return self
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

function base.variant:set_is_cuda(cuda)
  self.cuda = cuda
end

function base.variant:is_cuda()
  return self.cuda
end

function base.variant:set_is_external(external)
  self.external = external
end

function base.variant:is_external()
  return self.external
end

function base.variant:set_is_inline(inline)
  self.inline = inline
end

function base.variant:is_inline()
  return self.inline
end

do
  local global_kernel_id = 1
  function base.variant:add_cuda_kernel(kernel)
    if not self.cudakernels then
      self.cudakernels = {}
    end
    local kernel_id = global_kernel_id
    local kernel_name = self.task:get_name():concat("_") .. "_cuda" .. tostring(kernel_id)
    self.cudakernels[kernel_id] = {
      name = kernel_name,
      kernel = kernel,
    }
    global_kernel_id = global_kernel_id + 1
    return kernel_id
  end
end

function base.variant:get_cuda_kernels()
  return self.cudakernels or {}
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
    .. '_' .. self.task:get_task_id():asvalue()
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

local function make_task_wrapper(task_body)
  local return_type = task_body:gettype().returntype
  if return_type == terralib.types.unit then
    return terra(data : &opaque, datalen : c.size_t,
                 userdata : &opaque, userlen : c.size_t,
                 proc_id : c.legion_proc_id_t)
      var task : c.legion_task_t,
          regions : &c.legion_physical_region_t,
          num_regions : uint32,
          ctx : c.legion_context_t,
          runtime : c.legion_runtime_t
      c.legion_task_preamble(data, datalen, proc_id, &task, &regions, &num_regions, &ctx, &runtime)
      task_body(task, regions, num_regions, ctx, runtime)
      c.legion_task_postamble(runtime, ctx, nil, 0)
    end
  else
    return terra(data : &opaque, datalen : c.size_t,
                 userdata : &opaque, userlen : c.size_t,
                 proc_id : c.legion_proc_id_t)
      var task : c.legion_task_t,
          regions : &c.legion_physical_region_t,
          num_regions : uint32,
          ctx : c.legion_context_t,
          runtime : c.legion_runtime_t
      c.legion_task_preamble(data, datalen, proc_id, &task, &regions, &num_regions, &ctx, &runtime)
      var result = task_body(task, regions, num_regions, ctx, runtime)
      c.legion_task_postamble(runtime, ctx, result.value, result.size)
      c.free(result.value)
    end
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
      external = false,
      inline = false,
      cudakernels = false,
      config_options = false,
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

function base.task:set_field_id_param_symbols(t)
  assert(not self.field_id_param_symbols)
  assert(t)
  self.field_id_param_symbols = t
end

function base.task:get_field_id_param_symbols()
  assert(self.field_id_param_symbols)
  return self.field_id_param_symbols
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
  self.taskid = terralib.constant(c.legion_task_id_t, task_id)
end

function base.task:get_task_id()
  return self.taskid
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
end

function base.task:get_name()
  return self.name
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

function base.task:set_primary_variant(task)
  assert(not self.primary_variant)
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
  assert(not self.cuda_variant)
  self.cuda_variant = task
end

function base.task:get_cuda_variant()
  return self.cuda_variant
end

function base.task:set_parallel_task(task)
  assert(not self.parallel_task)
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
  return base.new_variant(self, name)
end

function base.task:add_complete_thunk(complete_thunk)
  assert(not self.is_complete)
  self.complete_thunks:insert(complete_thunk)
end

function base.task:complete()
  if not self.is_complete then
    self.is_complete = true
    for _, thunk in ipairs(self.complete_thunks) do
      thunk()
    end
  end
  return self
end

function base.task:compile()
  return self:complete()
end

function base.task:__tostring()
  return tostring(self:get_name())
end

do
  local next_task_id = base.initial_regent_task_id
  function base.new_task(name)
    if type(name) == "string" then
      name = data.newtuple(name)
    elseif data.is_tuple(name) then
      assert(data.all(name:map(function(n) return type(n) == "string" end)))
    else
      assert(false)
    end

    local task_id = next_task_id
    next_task_id = next_task_id + 1
    return setmetatable({
      name = name,
      taskid = terralib.constant(c.legion_task_id_t, task_id),
      variants = terralib.newlist(),
      calling_convention = false,

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
      complete_thunks = terralib.newlist(),
      is_complete = false,
    }, base.task)
  end
end

function base.is_task(x)
  return getmetatable(x) == base.task
end

return base
