-- Copyright 2016 Stanford University, NVIDIA Corporation
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

-- Bishop Code Generation

local ast = require("bishop/ast")
local log = require("bishop/log")
local std = require("bishop/std")
local regent_std = require("regent/std")

local c = terralib.includecstring [[
#include "legion_c.h"
#include "bishop_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]]

local codegen = {}

local property_setters = {
  task = {
    target = {}
  },
  region = {
    target = {}
  }
}

function property_setters:register_setter(rule_type, field, arg_type, setter)
  self[rule_type][field][arg_type] = setter
end

function property_setters:find_setter(rule_type, node)
  local setter_info = self[rule_type][node.field][node.value.expr_type]
  if not setter_info then
    log.error(node, "field " .. node.field .. " is not valid")
  end
  return setter_info
end

property_setters:register_setter("task", "target", std.processor_type,
                                 {fn = c.bishop_task_set_target_processor})
property_setters:register_setter("task", "target", std.processor_list_type,
                                 {fn = c.bishop_task_set_target_processor_list,
                                  cleanup = c.bishop_delete_processor_list})
property_setters:register_setter("region", "target", std.memory_type,
                                 {fn = c.bishop_region_set_target_memory})
property_setters:register_setter("region", "target", std.memory_list_type,
                                 {fn = c.bishop_region_set_target_memory_list,
                                  cleanup = c.bishop_delete_memory_list})

local processor_isa = {
  x86 = c.X86_ISA,
  cuda = c.CUDA_ISA,
}

local memory_kind = {
  global = c.GLOBAL_MEM,
  sysmem = c.SYSTEM_MEM,
  regmem = c.REGDMA_MEM,
  fbmem = c.GPU_FB_MEM,
  zcmem = c.Z_COPY_MEM,
  disk = c.DISK_MEM,
  hdf = c.HDF_MEM,
  file = c.FILE_MEM,
  l1cache = c.LEVEL1_CACHE,
  l2cache = c.LEVEL2_CACHE,
  l3cache = c.LEVEL3_CACHE,
}

function codegen.type(ty)
  if std.is_processor_type(ty) then
    return c.legion_processor_t
  elseif std.is_processor_list_type(ty) then
    return c.bishop_processor_list_t
  elseif std.is_memory_type(ty) then
    return c.legion_memory_t
  elseif std.is_memory_list_type(ty) then
    return c.bishop_memory_list_t
  elseif std.is_isa_type(ty) then
    return c.legion_processor_kind_t
  elseif std.is_memory_kind_type(ty) then
    return c.legion_memory_kind_t
  elseif std.is_point_type(ty) then
    return c.legion_domain_point_t
  elseif ty == int or ty == bool then
    return ty
  else
    assert(false, "type '" .. tostring(ty) .. "' should not be used for " ..
      "actual terra code generation")
  end
end

function codegen.expr(binders, state_var, node)
  local value = terralib.newsymbol(codegen.type(node.expr_type))
  local actions = quote
    var [value]
  end

  if node:is(ast.typed.expr.Keyword) then
    local keyword = node.value
    if keyword == "processors" then
      assert(std.is_processor_list_type(node.expr_type))
      actions = quote
        [actions]
        [value] = c.bishop_all_processors()
      end
      elseif keyword == "memories" then
      assert(std.is_memory_list_type(node.expr_type))
      actions = quote
        [actions]
        [value] = c.bishop_all_memories()
      end
    elseif std.is_isa_type(node.expr_type) then
      actions = quote
        [actions]
        [value] = [ processor_isa[keyword] ]
      end
    elseif std.is_memory_kind_type(node.expr_type) then
      actions = quote
        [actions]
        [value] = [ memory_kind[keyword] ]
      end
    else
      log.error(node, "keyword " .. keyword .. " is not yet supported")
    end

  elseif node:is(ast.typed.expr.Unary) then
    local rhs = codegen.expr(binders, state_var, node.rhs)
    actions = quote
      [rhs.actions]
      [actions]
      [value] = [std.quote_unary_op(node.op, rhs.value)]
    end

  elseif node:is(ast.typed.expr.Binary) then
    local lhs = codegen.expr(binders, state_var, node.lhs)
    local rhs = codegen.expr(binders, state_var, node.rhs)
    actions = quote
      [lhs.actions]
      [rhs.actions]
      [actions]
      [value] = [std.quote_binary_op(node.op, lhs.value, rhs.value)]
    end

  elseif node:is(ast.typed.expr.Ternary) then
    local cond = codegen.expr(binders, state_var, node.cond)
    local true_expr = codegen.expr(binders, state_var, node.true_expr)
    local false_expr = codegen.expr(binders, state_var, node.false_expr)
    actions = quote
      [actions]
      [cond.actions]
      if [cond.value] then
        [true_expr.actions]
        [value] = [true_expr.value]
      else
        [false_expr.actions]
        [value] = [false_expr.value]
      end
    end

  elseif node:is(ast.typed.expr.Filter) then
    local base = codegen.expr(binders, state_var, node.value)
    actions = quote
      [actions]
      [base.actions]
      [value] = [base.value]
    end
    node.constraints:map(function(constraint)
      assert(constraint:is(ast.typed.FilterConstraint))
      local v = codegen.expr(binders, state_var, constraint.value)
      if constraint.field == "isa" then
        assert(std.is_processor_list_type(node.value.expr_type))
        assert(std.is_isa_type(constraint.value.expr_type))
        actions = quote
          [actions]
          [v.actions]
          [value] = c.bishop_filter_processors_by_isa([value], [v.value])
          if [value].size == 0 then
            c.bishop_logger_debug("expression '%s' yields an empty list",
              [node:unparse()])
          end
          c.bishop_delete_processor_list([base.value])
        end

      elseif constraint.field == "kind" then
        assert(std.is_memory_list_type(node.value.expr_type))
        assert(std.is_memory_kind_type(constraint.value.expr_type))
        actions = quote
          [actions]
          [v.actions]
          [value] = c.bishop_filter_memories_by_kind([value], [v.value])
          if [value].size == 0 then
            c.bishop_logger_debug("expression '%s' yields an empty list",
              [node:unparse()])
          end
          c.bishop_delete_memory_list([base.value])
        end

      else
        assert(false, "unreachable")
      end
    end)

  elseif node:is(ast.typed.expr.Index) then
    local base = codegen.expr(binders, state_var, node.value)
    local index = codegen.expr(binders, state_var, node.index)
    if std.is_point_type(node.value.expr_type) then
      actions = quote
        [actions]
        [base.actions]
        [index.actions]
        [value] = [base.value].point_data[ [index.value] ]
      end
    else
      local cleanup
      if std.is_processor_list_type(node.value.expr_type) then
        cleanup = c.bishop_delete_processor_list
      else
        assert(std.is_memory_list_type(node.value_expr_type))
        cleanup = c.bishop_delete_memory_list
      end
      actions = quote
        [actions]
        [base.actions]
        [index.actions]
        do
          std.assert([index.value] >= 0 and [index.value] < [base.value].size,
            ["index " .. tostring(index.value) .. " is out-of-bounds"])
          [value] = [base.value].list[ [index.value] ]
        end
        [cleanup](base.value)
      end
    end
  elseif node:is(ast.typed.expr.Field) then
    if node.field == "memories" then
      local base = codegen.expr(binders, state_var, node.value)
      actions = quote
        [actions]
        [base.actions]
        do
          [value] = c.bishop_filter_memories_by_visibility([base.value])
          if [value].size == 0 then
            c.bishop_logger_debug("  expression '%s' yields an empty list",
              [node:unparse()])
          end
        end
      end
    elseif node.field == "size" then
      local base = codegen.expr(binders, state_var, node.value)
      actions = quote
        [actions]
        [base.actions]
        [value] = [base.value].size
      end

    else
      log.error(node, "field " .. node.field  ..  " is not supported")
    end
  elseif node:is(ast.typed.expr.Constant) then
    actions = quote
      [actions]
      [value] = [node.value]
    end
  elseif node:is(ast.typed.expr.Variable) then
    if binders[node.value] then
      actions = quote
        [actions]
        [value] = [ binders[node.value] ]
      end
    else 
      actions = quote
        [actions]
        [value] = [state_var].[node.value]
      end
    end
  elseif node:is(ast.typed.expr.Coerce) then
    if node.expr_type == int and std.is_point_type(node.value.expr_type) then
      local base = codegen.expr(binders, state_var, node.value)
      actions = quote
        [actions]
        [base.actions]
        [value] = [ base.value ].point_data[0]
      end
    else
      assert(false, "unknown coercion from type " ..
        tostring(node.value.expr_type) ..
        " to type " .. tostring(node.expr_type))
    end

  else
    assert(false, "unexpected node type: " .. tostring(node.node_type))
  end

  return {
    actions = actions,
    value = value,
  }
end

function codegen.property(binders, state_var, rule_type, obj_var, node)
  local setter_info = property_setters:find_setter(rule_type, node)
  local value = codegen.expr(binders, state_var, node.value)
  local actions = quote
    [value.actions]
    var result = [setter_info.fn]([obj_var], [value.value])
    if not result then
      c.bishop_logger_warning("  property '%s' was not properly assigned",
        node.field)
    end
  end
  if setter_info.cleanup then
    actions = quote
      [actions]
      [setter_info.cleanup]([value.value])
    end
  end
  return actions
end

function codegen.task_rule(state_type, node)
  local task_var = terralib.newsymbol(c.legion_task_t)
  local is_matched = terralib.newsymbol(bool)
  local point_var = terralib.newsymbol(c.legion_domain_point_t, "dp_")
  local selector = node.selector
  local position_string = node.position.filename .. ":" ..
    tostring(node.position.linenumber)
  assert(#selector.elements > 0)
  local first_element = selector.elements[1]
  assert(first_element:is(ast.typed.element.Task))
  local binders = {}
  local state_var = terralib.newsymbol(&state_type)

  local selector_body = quote var [is_matched] = true end
  if #first_element.name > 0 then
    assert(#first_element.name == 1)
    local task_name = first_element.name[1]
    if not rawget(_G, task_name) then
      log.error(first_element,
        "task '" .. task_name .. "' does not exist")
    end
    selector_body = quote
      [selector_body]
      var name = c.legion_task_get_name([task_var])
      [is_matched] = [is_matched] and (c.strcmp(name, [task_name]) == 0)
    end
  end

  local select_task_options_pattern_matches = quote end
  local select_target_for_point_pattern_matches = quote end
  local predicate_pattern_matches = quote end
  first_element.patterns:map(function(pattern)
    if pattern.field == "index" then
      local binder = terralib.newsymbol(c.legion_domain_point_t, pattern.binder)
      binders[pattern.binder] = binder
      select_task_options_pattern_matches = quote
        [select_task_options_pattern_matches]
        var [binder]
        [binder].dim = 1
        [binder].point_data[0] = 0
      end
      select_target_for_point_pattern_matches = quote
        [select_target_for_point_pattern_matches]
        var [binder]
        [binder] = [point_var]
      end
    else
      log.error(node, "field " .. pattern.field ..
      " is not supported for pattern match")
    end
  end)

  local select_task_options_body = quote end
  local select_target_for_point_body = quote end

  node.properties:map(function(property)
    if property.field == "target" then
      local value = codegen.expr(binders, state_var, property.value)
      select_target_for_point_body = quote
        [select_target_for_point_body]
        [value.actions]
      end
      if std.is_processor_list_type(property.value.expr_type) then
        select_target_for_point_body = quote
          [select_target_for_point_body]
          if [value.value].size == 0 then
            c.bishop_logger_warning(
              ["[slice_domain] expression at %s returned an empty list. " ..
               "this property might not get assigned properly!"],
              [property.value.position.filename .. ":" ..
               tostring(property.value.position.linenumber)])
            return c.bishop_get_no_processor()
          else
            var proc : c.legion_processor_t = [value.value].list[0]
            c.bishop_delete_processor_list([value.value])
            return proc
          end
        end
      elseif std.is_processor_type(property.value.expr_type) then
        select_target_for_point_body = quote
          [select_target_for_point_body]
          return [value.value]
        end
      else
        assert(false, "unreachable")
      end
      select_task_options_body = quote
        [select_task_options_body]
        [codegen.property(binders, state_var, "task", task_var, property)]
      end
    end
  end)

  local constraint_checks = quote end
  selector.constraints:map(function(constraint)
    local lhs = codegen.expr(binders, state_var, constraint.lhs)
    local rhs = codegen.expr(binders, state_var, constraint.rhs)
    constraint_checks = quote
      [constraint_checks]
      do
        [lhs.actions]
        [rhs.actions]
        [is_matched] = [is_matched] and [lhs.value] == [rhs.value]
      end
    end
  end)

  local terra matches(ptr        : &opaque,
                      [task_var])
    var [state_var] = [&state_type](ptr)
    [selector_body]
    [predicate_pattern_matches]
    [constraint_checks]
    if [is_matched] then
      c.bishop_logger_info("[slice_domain] rule at %s matches",
        position_string)
    else
      c.bishop_logger_info("[slice_domain] rule at %s was not applied",
        position_string)
    end
    return [is_matched]
  end

  local function early_out(callback)
    return quote
      if not [is_matched] then
        c.bishop_logger_info(["[" .. callback .. "] rule at %s was not applied"],
          position_string)
        return
      end
    end
  end

  local terra select_task_options(ptr        : &opaque,
                                  [task_var])
    var [state_var] = [&state_type](ptr)
    [selector_body]
    [early_out("select_task_options")]
    [select_task_options_pattern_matches]
    [constraint_checks]
    [early_out("select_task_options")]
    c.bishop_logger_info("[select_task_options] rule at %s matches",
      position_string)
    [select_task_options_body]
  end

  local terra select_target_for_point(ptr         : &opaque,
                                      [task_var],
                                      [point_var])
    var [state_var] = [&state_type](ptr)
    [select_target_for_point_pattern_matches]
    [select_target_for_point_body]
  end

  return {
    matches = matches,
    select_task_options = select_task_options,
    select_target_for_point = select_target_for_point,
    select_task_variant = 0,
  }
end

function codegen.region_rule(state_type, node)
  local task_var = terralib.newsymbol(c.legion_task_t)
  local req_var = terralib.newsymbol(c.legion_region_requirement_t)
  local req_idx = terralib.newsymbol(uint)
  local is_matched = terralib.newsymbol(bool, "is_matched")
  local selector_body = quote var [is_matched] = true end
  local selector = node.selector
  local position_string = node.position.filename .. ":" ..
    tostring(node.position.linenumber)
  assert(#selector.elements > 1)
  local first_element = selector.elements[1]
  assert(first_element:is(ast.typed.element.Region))
  local first_task_element = selector.elements[2]
  assert(first_task_element:is(ast.typed.element.Task))
  local binders = {}
  local state_var = terralib.newsymbol(&state_type)

  if #first_task_element.name > 0 then
    assert(#first_task_element.name == 1)
    local task_name = first_task_element.name[1]
    local regent_task = rawget(_G, task_name)
    if not regent_task then
      log.error(first_task_element,
        "task '" .. task_name .. "' does not exist")
    end
    selector_body = quote
      [selector_body]
      var name = c.legion_task_get_name([task_var])
      [is_matched] = [is_matched] and (c.strcmp(name, [task_name]) == 0)
    end
  end

  local pattern_matches = quote end
  first_task_element.patterns:map(function(pattern)
    if pattern.field == "target" then
      local binder = terralib.newsymbol(c.legion_processor_t, pattern.binder)
      binders[pattern.binder] = binder
      pattern_matches = quote
        [pattern_matches]
        var [binder]
        [binder] = c.legion_task_get_target_proc([task_var])
      end
    elseif pattern.field == "isa" then
      local binder = terralib.newsymbol(pattern.binder)
      binders[pattern.binder] = binder
      pattern_matches = quote
        [pattern_matches]
        var [binder]
        do
          var proc = c.legion_task_get_target_proc([task_var])
          [binder] = c.bishop_processor_get_isa(proc)
        end
      end
    else
      log.error(node, "field " .. pattern.field ..
      " is not supported for pattern match")
    end
  end)

  local constraint_checks = quote end
  selector.constraints:map(function(constraint)
    local lhs = codegen.expr(binders, state_var, constraint.lhs)
    local rhs = codegen.expr(binders, state_var, constraint.rhs)
    constraint_checks = quote
      [constraint_checks]
      do
        [lhs.actions]
        [rhs.actions]
        [is_matched] = [is_matched] and [lhs.value] == [rhs.value]
      end
    end
  end)

  local map_task_body = quote end
  if #first_element.name > 0 then
    if #first_task_element.name == 0 then
      log.error(first_task_element,
        "unnamed task element cannot have a named region element")
    end

    local param_name = "$" .. first_element.name[1]
    local task_name = first_task_element.name[1]
    local regent_task = rawget(_G, task_name)
    assert(regent_task, "unreachable")
    local task_params = regent_task.ast.params
    local param_type = nil
    local accum_idx, start_idx, end_idx = 0, 0, 0
    for _, param in pairs(task_params) do
      local param_type_in_signature = regent_std.as_read(param.param_type)
      if regent_std.is_region(param_type_in_signature) then
        local privileges =
          regent_std.find_task_privileges(param_type_in_signature,
                                   regent_task:getprivileges(),
                                   regent_task:get_coherence_modes(),
                                   regent_task:get_flags())
        if tostring(param.symbol) == param_name then
          param_type = param_type_in_signature
          start_idx = accum_idx
          end_idx = start_idx + #privileges - 1
          break
        end
        accum_idx = accum_idx + #privileges
      end
    end
    if not param_type then
      log.error(first_element,
        "parameter '" .. first_element.name[1] ..
        "' either does not exist or have a non-region type")
    end
    selector_body = quote
      [selector_body]
      [is_matched] = [is_matched] and
                     [req_idx] >= start_idx and
                     [req_idx] <= end_idx
    end
  end

  node.properties:map(function(property)
    map_task_body = quote
      [map_task_body]
      [codegen.property(binders, state_var, "region", req_var, property)]
    end
  end)

  local early_out = quote
    if not [is_matched] then
      c.bishop_logger_info("[map_task] rule at %s was not applied",
        position_string)
      return
    end
  end

  local terra map_task(ptr        : &opaque,
                       [task_var],
                       [req_var],
                       [req_idx])
    var [state_var] = [&state_type](ptr)
    [selector_body]
    [early_out]
    [pattern_matches]
    [constraint_checks]
    [early_out]
    c.bishop_logger_info("[map_task] rule at %s matches", position_string)
    [map_task_body]
  end

  return {
    pre_map_task = 0,
    map_task = map_task,
  }
end

function codegen.mapper_init(assignments)
  local entries = terralib.newlist()
  local binders = {}
  -- TODO: we might need to randomly generate this name with multiple mappers
  local mapper_state_type = terralib.types.newstruct("mapper_state")
  mapper_state_type.entries = assignments:map(function(assignment)
    return {
      field = assignment.binder,
      type = codegen.type(assignment.value.expr_type),
    }
  end)
  local mapper_state_var = terralib.newsymbol(&mapper_state_type)
  local mapper_init

  if sizeof(mapper_state_type) > 0 then
    terra mapper_init(ptr : &&opaque)
      @ptr = c.malloc([sizeof(mapper_state_type)])
      var [mapper_state_var] = [&mapper_state_type](@ptr)
      [assignments:map(function(assignment)
        local value = codegen.expr(binders, mapper_state_var, assignment.value)
        local mark_persistent = quote end
        if std.is_list_type(assignment.value.expr_type) then
          mark_persistent = quote [value.value].persistent = 1 end
        end
        return quote
          [value.actions]
          [mark_persistent]
          [mapper_state_var].[assignment.binder] = [value.value]
        end
      end)]
    end
  else
    terra mapper_init(ptr : &&opaque)
    end
  end
  return mapper_init, mapper_state_type
end

function codegen.mapper(node)
  local mapper_init, mapper_state_type =
    codegen.mapper_init(node.assignments)
  local task_rules =
    node.task_rules:map(function(rule)
      return codegen.task_rule(mapper_state_type, rule)
    end)
  local region_rules =
    node.region_rules:map(function(rule)
      return codegen.region_rule(mapper_state_type, rule)
    end)
  return {
    mapper_init = mapper_init,
    task_rules = task_rules,
    region_rules = region_rules,
  }
end

return codegen
