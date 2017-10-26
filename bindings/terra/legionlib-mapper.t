-- Copyright 2017 Stanford University
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

local legion_c = terralib.includec("legion.h")
local legion_terra = terralib.includecstring([[
#include "legion_terra.h"
#include "legion_terra_tasks.h"
]])
local std = terralib.includec("stdlib.h")

require 'legionlib-util'

local FALSE_RETURNED = 0
local TRUE_RETURNED = 1
local UNDEFINED = 2

-- Mapper wrapper for all Lua mappers

LuaMapperWrapper = {}
LuaMapperWrapper.__index = LuaMapperWrapper

local terra make_legion_default_mapper(ptr : &opaque)
  var mapper : legion_c.legion_default_mapper_t
  mapper.impl = ptr
  return mapper
end

function LuaMapperWrapper:new(cobj, child, machine_, runtime, local_proc_)
  local machine = Machine:from_cobj(machine_)
  local local_proc = Processor:from_cobj(coerce_processor(local_proc_, 0))
  local child_mapper = _G[child]:new(machine, runtime, local_proc)
  inherit(child_mapper, LuaMapperWrapper)
  self.cobj = make_legion_default_mapper(cobj)
  self.mapper = child_mapper
  self.machine = machine
  self.local_proc = local_proc
  self.local_kind = local_proc:kind()
  self.mapper.war_enabled = false
  self.machine_interface = MachineQueryInterface:new(machine)
end

local terra copy_domain_split(dst : &opaque,
                              idx : uint,
                              src : legion_c.legion_domain_split_t)
  [&legion_c.legion_domain_split_t](dst)[idx] = src
end

function LuaMapperWrapper:slice_domain_wrapper(task, domain_, slices)
  if not self.mapper.slice_domain then
    return UNDEFINED
  end
  local domain = Domain:from_cobj(coerce_domain(domain_, 0))
  self.mapper:slice_domain(task, domain, slices)
  return TRUE_RETURNED
end

local attrs_to_propagate =
  Array:new
  { "virtual_map", "early_map", "enable_WAR_optimization",
    "reduction_list", "make_persistent", "blocking_factor" }
local function set_req_attr(req, attr, value)
  return legion_c["legion_region_requirement_set_" .. attr](req, value);
end

local function propagate_back(req)
 local cobj = req.cobj
 attrs_to_propagate:itr(function(attr)
   set_req_attr(cobj, attr, req[attr])
 end)
 req.target_ranking:itr(function(mem)
   legion_c.legion_region_requirement_add_target_ranking(cobj, mem)
 end)
end

function LuaMapperWrapper:map_task_wrapper(task_)
  if not self.mapper.map_task then
    return UNDEFINED
  end
  local task = Task:from_cobj(task_)
  local result = self.mapper:map_task(task)
  task.regions:itr(propagate_back)
  if result then return TRUE_RETURNED else return FALSE_RETURNED end
end

function LuaMapperWrapper:map_inline_wrapper(inline_)
  if not self.mapper.map_inline then
    return UNDEFINED
  end
  local inline = Inline:from_cobj(inline_)
  local result = self.mapper:map_inline(inline)
  propagate_back(inline.requirement)
  if result then return TRUE_RETURNED else return FALSE_RETURNED end
end

function LuaMapperWrapper:notify_mapping_failed_wrapper(mappable)
  if not self.mapper.notify_mapping_failed then
    return UNDEFINED
  end
  self.mapper:notify_mapping_failed(mappable)
  return TRUE_RETURNED
end

local terra copy_processor(dst : &legion_c.legion_processor_t,
                           idx : uint,
                           src : legion_c.legion_processor_t)
  dst[idx] = src
end

function LuaMapperWrapper:decompose_index_space(domain, targets,
                                                splitting_factor,
                                                slices)
  do
    local targets_ =
      gcmalloc(legion_c.legion_processor_t, targets.size)
    for i = 0, targets.size - 1 do
      copy_processor(targets_, i, targets[i])
    end

    legion_terra.decompose_index_space(domain.cobj, targets_,
                                       targets.size,
                                       splitting_factor,
                                       slices)
  end
end

DefaultMapper = {}
DefaultMapper.__index = DefaultMapper

function DefaultMapper:map_task(child, task)
  local result =
    legion_c.legion_default_mapper_map_task(child.cobj, task)
  task.regions:itr(function(region)
    region:refresh_fields()
  end)
  return result
end

function DefaultMapper:map_inline(child, inline)
  local result =
    legion_c.legion_default_mapper_map_inline(child.cobj, inline)
  inline.requirement:refresh_fields()
  return result
end
