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
    local kernel_name = self.name:concat("_") .. "_cuda" .. tostring(kernel_id)
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

function base.variant:compile()
  self.task:complete()
  return self:get_definition():compile()
end

function base.variant:disas()
  self.task:complete()
  return self:get_definition():disas()
end

function base.variant:__tostring()
  return tostring(self:get_name())
end

do
  function base.new_variant(task, name)
    assert(base.is_task(task))

    if type(name) == "string" then
      name = data.newtuple(name)
    elseif data.is_tuple(name) then
      assert(data.all(name:map(function(n) return type(n) == "string" end)))
    else
      assert(false)
    end

    local variant = setmetatable({
      task = task,
      name = name,
      ast = false,
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
  if #self.variants > 0 then
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
