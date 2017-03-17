-- Copyright 2017 Stanford University, NVIDIA Corporation
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

local data = require("common/data")

local base = {}

-- #####################################
-- ## Legion Bindings
-- #################

terralib.linklibrary("liblegion_terra.so")
local c = terralib.includecstring([[
#include "legion_c.h"
#include "legion_terra.h"
#include "legion_terra_partitions.h"
#include <stdio.h>
#include <stdlib.h>
]])
base.c = c

-- #####################################
-- ## Tasks
-- #################

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
  self.param_symbols = symbols
end

function base.task:get_param_symbols()
  assert(self.param_symbols)
  return self.param_symbols
end

function base.task:set_params_struct(t)
  assert(not self.params_struct)
  self.params_struct = t
end

function base.task:get_params_struct()
  self:complete()
  assert(self.params_struct)
  return self.params_struct
end

function base.task:set_params_map_type(t)
  assert(not self.params_map_type)
  assert(t)
  self.params_map_type = t
end

function base.task:has_params_map_type()
  self:complete()
  return self.params_map_type
end

function base.task:get_params_map_type()
  self:complete()
  assert(self.params_map_type)
  return self.params_map_type
end

function base.task:set_params_map_label(label)
  assert(not self.params_map_label)
  assert(label)
  self.params_map_label = label
end

function base.task:has_params_map_label()
  self:complete()
  return self.params_map_label
end

function base.task:get_params_map_label()
  self:complete()
  assert(self.params_map_label)
  return self.params_map_label
end

function base.task:set_params_map_symbol(symbol)
  assert(not self.params_map_symbol)
  assert(symbol)
  self.params_map_symbol = symbol
end

function base.task:has_params_map_symbol()
  self:complete()
  return self.params_map_symbol
end

function base.task:get_params_map_symbol()
  self:complete()
  assert(self.params_map_symbol)
  return self.params_map_symbol
end

function base.task:set_field_id_param_labels(t)
  assert(not self.field_id_param_labels)
  self.field_id_param_labels = t
end

function base.task:get_field_id_param_labels()
  self:complete()
  assert(self.field_id_param_labels)
  return self.field_id_param_labels
end

function base.task:set_field_id_param_symbols(t)
  assert(not self.field_id_param_symbols)
  self.field_id_param_symbols = t
end

function base.task:get_field_id_param_symbols()
  self:complete()
  assert(self.field_id_param_symbols)
  return self.field_id_param_symbols
end

function base.task:setcuda(cuda)
  self.cuda = cuda
end

function base.task:getcuda()
  return self.cuda
end

function base.task:setexternal(external)
  self.external = external
end

function base.task:getexternal()
  return self.external
end

function base.task:setinline(inline)
  self.inline = inline
end

function base.task:getinline()
  return self.inline
end

local global_kernel_id = 1
function base.task:addcudakernel(kernel)
  if not self.cudakernels then
    self.cudakernels = {}
  end
  local kernel_id = global_kernel_id
  local kernel_name = self.name:concat("_") .. "_cuda" .. tostring(kernel_id)
  self.cudakernels[kernel_id] = {
    name = kernel_name,
    kernel = kernel,
  }
  global_kernel_id = global_kernel_id + 1
  return kernel_id
end

function base.task:getcudakernels()
  assert(self.cudakernels)
  return self.cudakernels
end

function base.task:settype(type, force)
  assert(force or not self.type)
  self.type = type
end

function base.task:gettype()
  assert(self.type)
  return self.type
end

function base.task:setprivileges(t)
  assert(not self.privileges)
  self.privileges = t
end

function base.task:getprivileges()
  assert(self.privileges)
  return self.privileges
end

function base.task:set_coherence_modes(t)
  assert(not self.coherence_modes)
  self.coherence_modes = t
end

function base.task:get_coherence_modes()
  assert(self.coherence_modes)
  return self.coherence_modes
end

function base.task:set_flags(t)
  assert(not self.flags)
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
  self.constraints = t
end

function base.task:get_constraints()
  assert(self.constraints)
  return self.constraints
end

function base.task:set_region_universe(t)
  assert(not self.region_universe)
  self.region_universe = t
end

function base.task:get_region_universe()
  assert(self.region_universe)
  return self.region_universe
end

function base.task:set_config_options(t)
  assert(not self.config_options)
  self.config_options = t
end

function base.task:get_config_options()
  self:complete()
  assert(self.config_options)
  return self.config_options
end

function base.task:gettaskid()
  return self.taskid
end

function base.task:settaskid(id)
  self.taskid = id
end

function base.task:getname()
  return self.name
end

function base.task:setname(name)
  if type(name) == "string" then
    name = data.newtuple(name)
  elseif data.is_tuple(name) then
    assert(data.all(name:map(function(n) return type(n) == "string" end)))
  else
    assert(false)
  end
  self.name = name
  if self:get_parallel_variant() then
    self:get_parallel_variant():setname(name .. data.newtuple("parallelized"))
  end
  if self:get_cuda_variant() then
    self:get_cuda_variant():setname(name)
  end
end

function base.task:getdefinition()
  self:complete()
  assert(self.definition)
  return self.definition
end

function base.task:setdefinition(definition)
  assert(not self.definition)
  self.definition = definition
end

function base.task:setast(ast)
  assert(not self.ast)
  self.ast = ast
end

function base.task:hasast()
  return self.ast
end

function base.task:getast()
  assert(self.ast)
  return self.ast
end

function base.task:is_variant_task()
  if self.source_variant then
    return true
  else
    return false
  end
end

function base.task:set_source_variant(source_variant)
  assert(not self.source_variant)
  self.source_variant = source_variant
end

function base.task:get_source_variant()
  assert(self.source_variant)
  return self.source_variant
end

function base.task:set_parallel_variant(task)
  self.parallel_variant = task
end

function base.task:get_parallel_variant()
  return self.parallel_variant
end

function base.task:set_cuda_variant(task)
  self.cuda_variant = task
end

function base.task:get_cuda_variant()
  return self.cuda_variant
end

function base.task:make_variant()
  local variant_task = base.newtask(self.name)
  variant_task:settaskid(self:gettaskid())
  variant_task:settype(self:gettype())
  variant_task:setprivileges(self:getprivileges())
  variant_task:set_coherence_modes(self:get_coherence_modes())
  variant_task:set_conditions(self:get_conditions())
  variant_task:set_param_constraints(self:get_param_constraints())
  variant_task:set_flags(self:get_flags())
  variant_task:set_constraints(self:get_constraints())
  variant_task:set_source_variant(self)
  return variant_task
end

function base.task:set_complete_thunk(complete_thunk)
  assert(not self.complete_thunk)
  self.complete_thunk = complete_thunk
end

function base.task:complete()
  assert(self.complete_thunk)
  if not self.is_complete then
    self.is_complete = true
    return self.complete_thunk()
  end
end

function base.task:compile()
  return self:getdefinition():compile()
end

function base.task:disas()
  return self:getdefinition():disas()
end

function base.task:__tostring()
  return tostring(self:getname())
end

do
  local next_task_id = 1
  function base.newtask(name)
    assert(data.is_tuple(name))
    local task_id = next_task_id
    next_task_id = next_task_id + 1
    return setmetatable({
      name = name,
      taskid = terralib.constant(c.legion_task_id_t, task_id),
      ast = false,
      definition = false,
      cuda = false,
      external = false,
      inline = false,
      cudakernels = false,
      param_symbols = false,
      params_struct = false,
      params_map_type = false,
      params_map_label = false,
      params_map_symbol = false,
      field_id_param_labels = false,
      field_id_param_symbols = false,
      type = false,
      privileges = false,
      coherence_modes = false,
      flags = false,
      conditions = false,
      param_constraints = false,
      constraints = false,
      region_universe = false,
      config_options = false,
      source_variant = false,
      complete_thunk = false,
      is_complete = false,
      parallel_variant = false,
      cuda_variant = false,
    }, base.task)
  end
end

function base.is_task(x)
  return getmetatable(x) == base.task
end

return base
