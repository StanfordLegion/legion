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
local regex = require("bishop/regex")
local regent_std = require("regent/std")
local data = require("regent/data")

local c = terralib.includecstring [[
#include "legion_c.h"
#include "bishop_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]]

local codegen = {}

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

--function codegen.task_rule(state_type, node)
--  local rt_var = terralib.newsymbol(c.legion_mapper_runtime_t)
--  local ctx_var = terralib.newsymbol(c.legion_mapper_context_t)
--  local task_var = terralib.newsymbol(c.legion_task_t)
--  local options_var = terralib.newsymbol(c.legion_task_options_t)
--  local map_task_input_var = terralib.newsymbol(c.legion_map_task_input_t)
--  local map_task_output_var = terralib.newsymbol(c.legion_map_task_output_t)
--  local is_matched = terralib.newsymbol(bool)
--  local point_var = terralib.newsymbol(c.legion_domain_point_t, "dp_")
--  local selector = node.selector
--  local position_string = node.position.filename .. ":" ..
--    tostring(node.position.linenumber)
--  assert(#selector.elements > 0)
--  local first_element = selector.elements[1]
--  assert(first_element:is(ast.typed.element.Task))
--  local binders = {}
--  local state_var = terralib.newsymbol(&state_type)
--
--  local selector_body = quote var [is_matched] = true end
--  if #first_element.name > 0 then
--    assert(#first_element.name == 1)
--    local task_name = first_element.name[1]
--    if not rawget(_G, task_name) then
--      log.error(first_element,
--        "task '" .. task_name .. "' does not exist")
--    end
--    selector_body = quote
--      [selector_body]
--      var name = c.legion_task_get_name([task_var])
--      [is_matched] = [is_matched] and (c.strcmp(name, [task_name]) == 0)
--    end
--  end
--
--  local select_task_options_pattern_matches = quote end
--  local map_task_pattern_matches = quote end
--  local predicate_pattern_matches = quote end
--  first_element.patterns:map(function(pattern)
--    if pattern.field == "index" then
--      local binder = terralib.newsymbol(c.legion_domain_point_t, pattern.binder)
--      binders[pattern.binder] = binder
--      select_task_options_pattern_matches = quote
--        [select_task_options_pattern_matches]
--        var [binder]
--        [binder].dim = 1
--        [binder].point_data[0] = 0
--      end
--      map_task_pattern_matches = quote
--        [map_task_pattern_matches]
--        var [binder] = c.legion_task_get_index_point([task_var])
--      end
--    else
--      log.error(node, "field " .. pattern.field ..
--      " is not supported for pattern match")
--    end
--  end)
--
--  local select_task_options_body = quote end
--  local map_task_body = quote end
--
--  node.properties:map(function(property)
--    if property.field == "target" then
--      local value = codegen.expr(binders, state_var, property.value)
--      if std.is_processor_list_type(property.value.expr_type) then
--        local result = terralib.newsymbol(c.legion_processor_t)
--        value.actions = quote
--          [value.actions]
--          var [result]
--          if [value.value].size > 0 then
--            [result] = [value.value].list[0]
--          else
--            [result] = c.bishop_get_no_processor()
--          end
--        end
--        value.value = result
--      end
--
--      select_task_options_body = quote
--        [select_task_options_body]
--        [value.actions]
--        [options_var].initial_proc = [value.value]
--      end
--      map_task_body = quote
--        [map_task_body]
--        [value.actions]
--        c.legion_map_task_output_target_procs_clear([map_task_output_var])
--        c.legion_map_task_output_target_procs_add([map_task_output_var], [value.value])
--      end
--
--    elseif property.field == "priority" then
--      local value = codegen.expr(binders, state_var, property.value)
--      map_task_body = quote
--        [map_task_body]
--        [value.actions]
--        c.legion_map_task_output_task_priority_set([map_task_output_var], [value.value])
--      end
--    else
--      assert(false, "unsupported")
--    end
--  end)
--
--  local constraint_checks = quote end
--  selector.constraints:map(function(constraint)
--    local lhs = codegen.expr(binders, state_var, constraint.lhs)
--    local rhs = codegen.expr(binders, state_var, constraint.rhs)
--    constraint_checks = quote
--      [constraint_checks]
--      do
--        [lhs.actions]
--        [rhs.actions]
--        [is_matched] = [is_matched] and [lhs.value] == [rhs.value]
--      end
--    end
--  end)
--
--  local terra matches(ptr        : &opaque,
--                      [task_var])
--    var [state_var] = [&state_type](ptr)
--    [selector_body]
--    [predicate_pattern_matches]
--    [constraint_checks]
--    if [is_matched] then
--      c.bishop_logger_info("[slice_domain] rule at %s matches",
--        position_string)
--    else
--      c.bishop_logger_info("[slice_domain] rule at %s was not applied",
--        position_string)
--    end
--    return [is_matched]
--  end
--
--  local function early_out(callback)
--    return quote
--      if not [is_matched] then
--        c.bishop_logger_info(["[" .. callback .. "] rule at %s was not applied"],
--          position_string)
--        return
--      end
--    end
--  end
--
--  local terra select_task_options(ptr        : &opaque,
--                                  [rt_var],
--                                  [ctx_var],
--                                  [task_var],
--                                  [options_var])
--    var [state_var] = [&state_type](ptr)
--    [selector_body]
--    [early_out("select_task_options")]
--    [select_task_options_pattern_matches]
--    [constraint_checks]
--    [early_out("select_task_options")]
--    c.bishop_logger_info("[select_task_options] rule at %s matches",
--      position_string)
--    [select_task_options_body]
--  end
--
--  local terra map_task(ptr        : &opaque,
--                       [rt_var],
--                       [ctx_var],
--                       [task_var],
--                       [map_task_input_var],
--                       [map_task_output_var])
--    var [state_var] = [&state_type](ptr)
--    [selector_body]
--    [early_out("map_task")]
--    [map_task_pattern_matches]
--    [constraint_checks]
--    [early_out("map_task")]
--    c.bishop_logger_info("[map_task] rule at %s matches", position_string)
--    [map_task_body]
--  end
--
--  return {
--    matches = matches,
--    select_task_options = select_task_options,
--    map_task = map_task,
--  }
--end
--
--function codegen.region_rule(state_type, node)
--  local rt_var = terralib.newsymbol(c.legion_mapper_runtime_t)
--  local ctx_var = terralib.newsymbol(c.legion_mapper_context_t)
--  local task_var = terralib.newsymbol(c.legion_task_t)
--  local map_task_input_var = terralib.newsymbol(c.legion_map_task_input_t)
--  local map_task_output_var = terralib.newsymbol(c.legion_map_task_output_t)
--  local is_matched = terralib.newsymbol(bool, "is_matched")
--  local selector_body = quote var [is_matched] = true end
--  local selector = node.selector
--  local position_string = node.position.filename .. ":" ..
--    tostring(node.position.linenumber)
--  assert(#selector.elements > 1)
--  local first_element = selector.elements[1]
--  assert(first_element:is(ast.typed.element.Region))
--  local first_task_element = selector.elements[2]
--  assert(first_task_element:is(ast.typed.element.Task))
--  local binders = {}
--  local state_var = terralib.newsymbol(&state_type)
--
--  if #first_task_element.name > 0 then
--    assert(#first_task_element.name == 1)
--    local task_name = first_task_element.name[1]
--    local regent_task = rawget(_G, task_name)
--    if not regent_task then
--      log.error(first_task_element,
--        "task '" .. task_name .. "' does not exist")
--    end
--    selector_body = quote
--      [selector_body]
--      var name = c.legion_task_get_name([task_var])
--      [is_matched] = [is_matched] and (c.strcmp(name, [task_name]) == 0)
--    end
--  end
--
--  local pattern_matches = quote end
--  first_task_element.patterns:map(function(pattern)
--    if pattern.field == "target" then
--      local binder = terralib.newsymbol(c.legion_processor_t, pattern.binder)
--      binders[pattern.binder] = binder
--      pattern_matches = quote
--        [pattern_matches]
--        var [binder]
--        [binder] = c.legion_task_get_target_proc([task_var])
--      end
--    elseif pattern.field == "isa" then
--      local binder = terralib.newsymbol(pattern.binder)
--      binders[pattern.binder] = binder
--      pattern_matches = quote
--        [pattern_matches]
--        var [binder]
--        do
--          var proc = c.legion_task_get_target_proc([task_var])
--          [binder] = c.bishop_processor_get_isa(proc)
--        end
--      end
--    else
--      log.error(node, "field " .. pattern.field ..
--      " is not supported for pattern match")
--    end
--  end)
--
--  local constraint_checks = quote end
--  selector.constraints:map(function(constraint)
--    local lhs = codegen.expr(binders, state_var, constraint.lhs)
--    local rhs = codegen.expr(binders, state_var, constraint.rhs)
--    constraint_checks = quote
--      [constraint_checks]
--      do
--        [lhs.actions]
--        [rhs.actions]
--        [is_matched] = [is_matched] and [lhs.value] == [rhs.value]
--      end
--    end
--  end)
--
--  local map_task_body = quote end
--  local start_idx_var = terralib.newsymbol(int)
--  local end_idx_var = terralib.newsymbol(int)
--  local idx_assignments = terralib.newlist()
--
--  if #first_element.name > 0 then
--    if #first_task_element.name == 0 then
--      log.error(first_task_element,
--        "unnamed task element cannot have a named region element")
--    end
--
--    local param_name = first_element.name[1]
--    local task_name = first_task_element.name[1]
--    local regent_task = rawget(_G, task_name)
--    assert(regent_task, "unreachable")
--    local task_params = regent_task.ast.params
--    local param_type = nil
--    local accum_idx, start_idx, end_idx = 0, 0, 0
--    for _, param in pairs(task_params) do
--      local param_type_in_signature = regent_std.as_read(param.param_type)
--      if regent_std.is_region(param_type_in_signature) then
--        local privileges =
--          regent_std.find_task_privileges(param_type_in_signature,
--                                   regent_task:getprivileges(),
--                                   regent_task:get_coherence_modes(),
--                                   regent_task:get_flags())
--        if param.symbol:hasname() == param_name then
--          param_type = param_type_in_signature
--          start_idx = accum_idx
--          end_idx = start_idx + #privileges
--        end
--        accum_idx = accum_idx + #privileges
--      end
--    end
--    if not param_type then
--      log.error(first_element,
--        "parameter '" .. first_element.name[1] ..
--        "' either does not exist or have a non-region type")
--    end
--    idx_assignments = quote
--      [start_idx_var] = start_idx
--      [end_idx_var] = end_idx
--    end
--  else
--    idx_assignments = quote
--      [start_idx_var] = 0
--      [end_idx_var] = c.legion_task_get_regions_size([task_var])
--    end
--  end
--
--  node.properties:map(function(property)
--    if property.field == "target" then
--      local value = codegen.expr(binders, state_var, property.value)
--      if std.is_memory_list_type(property.value.expr_type) then
--        local result = terralib.newsymbol(c.legion_memory_t)
--        value.actions = quote
--          [value.actions]
--          var [result]
--          if [value.value].size > 0 then
--            [result] = [value.value].list[0]
--          else
--            [result] = c.bishop_get_no_memory()
--          end
--        end
--        value.value = result
--      end
--
--      map_task_body = quote
--        [map_task_body]
--        [value.actions]
--        do
--          var [start_idx_var], [end_idx_var]
--          [idx_assignments]
--          for idx = [start_idx_var], [end_idx_var] do
--            var req : c.legion_region_requirement_t =
--              c.legion_task_get_region([task_var], idx)
--            var fields_size =
--              c.legion_region_requirement_get_instance_fields_size(req)
--            var fields : &c.legion_field_id_t =
--              [&c.legion_field_id_t](
--                c.malloc([sizeof(c.legion_field_id_t)] * fields_size))
--            c.legion_region_requirement_get_instance_fields(req, fields,
--              fields_size)
--
--            var layout = c.legion_layout_constraint_set_create()
--            var priv = c.legion_region_requirement_get_privilege(req)
--            var redop = c.legion_region_requirement_get_redop(req)
--            c.legion_layout_constraint_set_add_memory_constraint(
--              layout,
--              c.legion_memory_kind([value.value]))
--            c.legion_layout_constraint_set_add_field_constraint(
--              layout, fields, fields_size, false, false)
--
--            if priv == c.REDUCE then
--              c.legion_layout_constraint_set_add_specialized_constraint(
--                layout, c.REDUCTION_FOLD_SPECIALIZE, redop)
--            elseif priv ~= c.NO_ACCESS then
--              var dims : uint[4]
--              dims[0], dims[1], dims[2], dims[3] =
--                c.DIM_X, c.DIM_Y, c.DIM_Z, c.DIM_F
--              c.legion_layout_constraint_set_add_ordering_constraint(
--                layout, dims, 4, false)
--            end
--
--            var region = c.legion_region_requirement_get_region(req)
--            var inst : c.legion_physical_instance_t
--            var created : bool
--            var success =
--              c.legion_mapper_runtime_find_or_create_physical_instance_layout_constraint(
--                [rt_var], [ctx_var], [value.value],
--                layout, &region, 1, &inst, &created, true, 0, true)
--            std.assert(success, "instance creation should succeed")
--
--            c.legion_map_task_output_chosen_instances_set(
--              [map_task_output_var], idx, &inst, 1)
--
--            c.legion_physical_instance_destroy(inst)
--            c.legion_layout_constraint_set_destroy(layout)
--            c.free(fields)
--          end
--        end
--      end
--    else
--      assert(false, "unsupported")
--    end
--  end)
--
--  local early_out = quote
--    if not [is_matched] then
--      c.bishop_logger_info("[map_task] rule at %s was not applied",
--        position_string)
--      return
--    end
--  end
--
--  local terra map_task(ptr        : &opaque,
--                       [rt_var],
--                       [ctx_var],
--                       [task_var],
--                       [map_task_input_var],
--                       [map_task_output_var])
--    var [state_var] = [&state_type](ptr)
--    [selector_body]
--    [early_out]
--    [pattern_matches]
--    [constraint_checks]
--    [early_out]
--    c.bishop_logger_info("[map_task] rule at %s matches", position_string)
--    [map_task_body]
--  end
--
--  return {
--    pre_map_task = 0,
--    map_task = map_task,
--  }
--end

local function expr_constant(c)
  return ast.typed.expr.Constant {
    value = c,
    position = ast.trivial_pos(),
    expr_type = int,
  }
end

local function expr_keyword(keyword, ty)
  return ast.typed.expr.Keyword {
    value = keyword,
    position = ast.trivial_pos(),
    expr_type = ty,
  }
end

local default_task_properties = {
  priority = expr_constant(0),
  target = ast.typed.expr.Filter {
    value = expr_keyword("processors", std.processor_list_type),
    constraints = terralib.newlist {
      ast.typed.FilterConstraint {
        field = "isa",
        value = expr_keyword("x86", std.isa_type),
        position = ast.trivial_pos(),
      }
    },
    expr_type = std.processor_list_type,
    position = ast.trivial_pos(),
  },
}

local function merge_task_properties(rules)
  local properties = {}
  for key, value in pairs(default_task_properties) do
    properties[key] = value
  end
  for idx = 1, #rules do
    for _, property in pairs(rules[idx].properties) do
      properties[property.field] = property.value
    end
  end
  return properties
end

local default_region_properties = {
  create = expr_keyword("allow", std.compile_option_type),
  target = ast.typed.expr.Filter {
    value = expr_keyword("memories", std.memory_list_type),
    constraints = terralib.newlist {
      ast.typed.FilterConstraint {
        field = "kind",
        value = expr_keyword("sysmem", std.memory_kind_type),
        position = ast.trivial_pos(),
      }
    },
    expr_type = std.memory_list_type,
    position = ast.trivial_pos(),
  },
}

local function merge_region_properties(rules)
  local properties = {}
  for key, value in pairs(default_region_properties) do
    properties[key] = value
  end
  for idx = 1, #rules do
    for _, property in pairs(rules[idx].properties) do
      properties[property.field] = property.value
    end
  end
  return properties
end

local function tostring_selectors(rules)
  if #rules == 0 then return "default policy" end
  local str = rules[1].selector:unparse()
  for idx = 2, #rules do
    str = str .. ", " .. rules[idx].selector:unparse()
  end
  return str
end

function codegen.pattern_match(binders, task_var, elem)
  local actions = quote end
  elem.patterns:map(function(pattern)
    if pattern.field == "index" then
      local binder =
        terralib.newsymbol(c.legion_domain_point_t, pattern.binder)
      binders[pattern.binder] = binder
      actions = quote
        [actions]
        var [binder] = c.legion_task_get_index_point([task_var])
      end
    else
      log.error(elem, "field " .. pattern.field ..
      " is not supported for pattern match")
    end
  end)
  return actions
end

function codegen.select_task_options(rules, automata, signatures,
                                     mapper_state_type)
  local rt_var = terralib.newsymbol(c.legion_mapper_runtime_t)
  local ctx_var = terralib.newsymbol(c.legion_mapper_context_t)
  local task_var = terralib.newsymbol(c.legion_task_t)
  local options_var = terralib.newsymbol(c.legion_task_options_t)
  local state_var = terralib.newsymbol(&mapper_state_type)

  local binders = {}
  local task_rules =
    data.filter(function(rule) return rule.rule_type == "task" end, rules)
  local last_elems =
    task_rules:map(function(rule)
      return rule.selector.elements[#rule.selector.elements]
    end)
  -- TODO: handle binder naming collision
  local body = quote
    [last_elems:map(std.curry2(codegen.pattern_match, binders, task_var))]
  end

  local properties = merge_task_properties(task_rules)
  for key, value_ast in pairs(properties) do
    if key == "target" then
      local value = codegen.expr(binders, state_var, value_ast)
      if std.is_processor_list_type(value_ast.expr_type) then
        local result = terralib.newsymbol(c.legion_processor_t)
        value.actions = quote
          [value.actions]
          var [result]
          if [value.value].size > 0 then
            [result] = [value.value].list[0]
          else
            [result] = c.bishop_get_no_processor()
          end
        end
        value.value = result
      end
      body = quote
        [body]
        [value.actions]
        [options_var].initial_proc = [value.value]
      end
    end
  end

  local selector_summary =
    terralib.constant(rawstring, tostring_selectors(rules))
  return terra(ptr : &opaque, [rt_var], [ctx_var], [task_var], [options_var])
    c.bishop_logger_debug("[select_task_options] merged from %s",
                          selector_summary)
    var [state_var] = [&mapper_state_type](ptr)
    -- XXX: These should be handled in the same way as other properties
    [options_var].inline_task = false
    [options_var].map_locally = true
    [options_var].stealable = false
    [body]
  end
end

function codegen.map_task(rules, automata, signatures, mapper_state_type)
  local rt_var = terralib.newsymbol(c.legion_mapper_runtime_t)
  local ctx_var = terralib.newsymbol(c.legion_mapper_context_t)
  local task_var = terralib.newsymbol(c.legion_task_t)
  local map_task_input_var = terralib.newsymbol(c.legion_map_task_input_t)
  local map_task_output_var = terralib.newsymbol(c.legion_map_task_output_t)
  local state_var = terralib.newsymbol(&mapper_state_type)
  local body = quote end

  local binders = {}
  local task_rules =
    data.filter(function(rule) return rule.rule_type == "task" end, rules)
  local region_rules =
    data.filter(function(rule) return rule.rule_type == "region" end, rules)
  local last_elems =
    rules:map(function(rule)
      if rule.rule_type == "task" then
        return rule.selector.elements[#rule.selector.elements]
      elseif rule.rule_type == "region" then
        return rule.selector.elements[#rule.selector.elements - 1]
      else
        assert(false, "unreachable")
      end
    end)
  -- TODO: handle binder naming collision
  -- TODO: handle pattern match on tasks differently than that on regions
  local body = quote
    [last_elems:map(std.curry2(codegen.pattern_match, binders, task_var))]
  end

  local task_properties = merge_task_properties(task_rules)
  local region_properties = merge_task_properties(region_rules)
  for key, value_ast in pairs(task_properties) do
    local value = codegen.expr(binders, state_var, value_ast)
    if key == "target" then
      if std.is_processor_list_type(value_ast.expr_type) then
        local result = terralib.newsymbol(c.legion_processor_t)
        value.actions = quote
          [value.actions]
          var [result]
          if [value.value].size > 0 then
            [result] = [value.value].list[0]
          else
            [result] = c.bishop_get_no_processor()
          end
        end
        value.value = result
      end
      body = quote
        [body]
        [value.actions]
        c.legion_map_task_output_target_procs_clear([map_task_output_var])
        c.legion_map_task_output_target_procs_add([map_task_output_var],
                                                  [value.value])
      end
    elseif key == "priority" then
      body = quote
        [body]
        [value.actions]
        c.legion_map_task_output_task_priority_set([map_task_output_var],
                                                   [value.value])
      end
    end
  end

  local selector_summary =
    terralib.constant(rawstring, tostring_selectors(rules))
  return terra(ptr : &opaque, [rt_var], [ctx_var], [task_var],
               [map_task_input_var], [map_task_output_var])
    c.bishop_logger_debug("[map_task] merged from %s",
                          selector_summary)
    var [state_var] = [&mapper_state_type](ptr)
    [body]
  end
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

local function hash_tags(tags)
  local tbl = {}
  for tag, _ in pairs(tags) do
    tbl[#tbl + 1] = tag
  end
  table.sort(tbl)
  local str = ""
  for idx = 1, #tbl do
    str = str .. "-" .. tbl[idx]
  end
  return str
end

function codegen.rules(rules, automata, signatures, mapper_state_type)
  local state_to_mapper_impl_id = {}
  local next_mapper_impl_id = 0
  local rule_hash_to_mapper_impl_id = {}
  local mapper_impl_id_to_mapper_impl = terralib.newlist()

  local function add_mapper_impl(hash, state_id, impl_id, impl)
    rule_hash_to_mapper_impl_id[hash] = impl_id
    state_to_mapper_impl_id[state_id] = impl_id
    mapper_impl_id_to_mapper_impl[impl_id] = impl
  end

  for state, _ in pairs(automata.states) do
    if automata.final[state] then
      local hash = hash_tags(state.tags)
      if not rule_hash_to_mapper_impl_id[hash] then
        local selected_rules = terralib.newlist()
        for idx = 1, #rules do
          if state.tags[idx] then
            selected_rules:insert(rules[idx])
          end
        end
        local select_task_options =
          codegen.select_task_options(selected_rules, automata, signatures,
                                      mapper_state_type)
        local map_task =
          codegen.map_task(selected_rules, automata, signatures,
                           mapper_state_type)
        --print(tostring_selectors(selected_rules))
        --map_task:printpretty()
        add_mapper_impl(hash, state.id, next_mapper_impl_id, {
          select_task_options = select_task_options,
          map_task = map_task,
        })
        next_mapper_impl_id = next_mapper_impl_id + 1
      else
        local mapper_impl_id = rule_hash_to_mapper_impl_id[hash]
        state_to_mapper_impl_id[state.id] = mapper_impl_id
      end
    end
  end

  -- generate default implementations for states with no matching rules
  local empty_list = terralib.newlist()
  local select_task_options =
    codegen.select_task_options(empty_list, automata, signatures,
                                mapper_state_type)
  local map_task =
    codegen.map_task(empty_list, automata, signatures,
                     mapper_state_type)
  for state, _ in pairs(automata.states) do
    if not automata.final[state] then
      add_mapper_impl("", state.id, next_mapper_impl_id, {
        select_task_options = select_task_options,
        map_task = map_task,
      })
    end
  end

  return state_to_mapper_impl_id, mapper_impl_id_to_mapper_impl
end

function codegen.automata(automata)
  local all_symbols = automata.all_symbols
  local state_to_transition_impl = terralib.newlist()
  for state, _ in pairs(automata.states) do
    local state_var = terralib.newsymbol(&opaque)
    local task_var = terralib.newsymbol(c.legion_task_t)
    local result_var = terralib.newsymbol(c.legion_task_id_t)
    local body = quote end
    for sym, next_state in pairs(state.trans) do
      local symbol = all_symbols[sym]
      if symbol:is(regex.symbol.TaskId) then
        body = quote
          [body]
          if c.legion_task_get_task_id([task_var]) == [sym] then
            return [next_state.id]
          end
        end
      elseif symbol:is(regex.symbol.Constraint) then
        -- TODO: handle constraints from unification
        local binders = {}
        local value = codegen.expr(binders, state_var, symbol.constraint.value)
        if symbol.constraint.field == "isa" then
          body = quote
            [body]
            do
              [value.actions]
              var proc = c.legion_task_get_target_proc([task_var])
              if proc.id ~= c.NO_PROC.id and
                 c.bishop_processor_get_isa(proc) == [value.value] then
                  return [next_state.id]
              end
            end
          end
        else
          assert(false, "not supported yet")
        end
      else
        assert(false, "not supported yet")
      end
    end
    body = quote
      [body]
      return [state.id]
    end
    state_to_transition_impl[state.id] =
      terra([state_var], [task_var]) : c.bishop_matching_state_t
        [body]
      end
  end
  return state_to_transition_impl
end

function codegen.mapper(node)
  local mapper_init, mapper_state_type =
    codegen.mapper_init(node.assignments)

  local state_to_mapper_impl_id, mapper_impl_id_to_mapper_impl =
    codegen.rules(node.rules, node.automata, node.task_signatures,
                  mapper_state_type)

  local state_to_transition_impl =
    codegen.automata(node.automata)

  return {
    mapper_init = mapper_init,
    state_to_mapper_impl_id = state_to_mapper_impl_id,
    state_to_transition_impl = state_to_transition_impl,
    mapper_impl_id_to_mapper_impl = mapper_impl_id_to_mapper_impl,
  }
end

return codegen
