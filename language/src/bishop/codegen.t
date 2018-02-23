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

-- Bishop Code Generation

local ast = require("bishop/ast")
local log = require("bishop/log")
local std = require("bishop/std")
local regex = require("bishop/regex")
local data = require("common/data")
local regent_std = require("regent/std")
local regent_codegen_hooks = require("regent/codegen_hooks")

local c = terralib.includecstring [[
#include "legion.h"
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

local processor_kind = {
  loc = c.LOC_PROC,
  toc = c.TOC_PROC,
  io = c.IO_PROC,
  openmp = c.OMP_PROC,
  util = c.UTIL_PROC,
  group = c.PROC_GROUP,
  set = c.PROC_SET,
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

local layout_field_name = "__layouts"
local instance_field_name = "__instances"
local slice_field_name = "__slices"

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
  elseif std.is_processor_kind_type(ty) then
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
    elseif std.is_processor_kind_type(node.expr_type) then
      actions = quote
        [actions]
        [value] = [ processor_kind[keyword] ]
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
        if std.is_processor_list_type(node.value.expr_type) then
          assert(std.is_processor_kind_type(constraint.value.expr_type))
          actions = quote
            [actions]
            [v.actions]
            [value] = c.bishop_filter_processors_by_kind([value], [v.value])
            if [value].size == 0 then
              c.bishop_logger_debug("expression '%s' yields an empty list",
                [node:unparse()])
            end
            c.bishop_delete_processor_list([base.value])
          end
        elseif std.is_memory_list_type(node.value.expr_type) then
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
          assert(false, "unsupported")
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

local default_task_target_binder = std.newsymbol()

local default_region_pattern_matches = terralib.newlist({
  ast.typed.PatternMatch {
    field = "target",
    binder = default_task_target_binder,
    position = ast.trivial_pos(),
  }
})

local default_region_properties = {
  create = expr_keyword("allow", std.compile_option_type),
  target = ast.typed.expr.Filter {
    value = ast.typed.expr.Field {
      value = ast.typed.expr.Variable {
        value = default_task_target_binder,
        expr_type = std.processor_type,
        position = ast.trivial_pos(),
      },
      field = "memories",
      expr_type = std.memory_list_type,
      position = ast.trivial_pos(),
    },
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

local function merge_region_properties(rules, signature)
  local all_properties = terralib.newlist()
  local num_reqs = 0
  if signature.reqs then num_reqs = #signature.reqs end
  for idx = 1, num_reqs do
    local properties = {}
    for key, value in pairs(default_region_properties) do
      properties[key] = value(value)
    end
    if regent_std.is_reduction_op(signature.reqs[idx].privilege) then
      properties.create.value = "demand"
    end
    for ridx = 1, #rules do
      local rule = rules[ridx]
      local region_elem =
        rule.selector.elements[#rule.selector.elements]
      local matched = #region_elem.name == 0
      region_elem.name:map(function(name)
        matched = matched or
          signature.region_params[name][idx]
      end)
      if matched then
        for _, property in pairs(rules[ridx].properties) do
          properties[property.field] = property.value(property.value)
        end
      end
    end
    all_properties:insert(properties)
  end
  return all_properties
end

local function tostring_selectors(rules)
  if #rules == 0 then return "default policy" end
  local str = rules[1].selector:unparse()
  for idx = 2, #rules do
    str = str .. ", " .. rules[idx].selector:unparse()
  end
  return str
end

function codegen.pattern_match(binders, task_var, pattern)
  local actions = quote end
  if pattern.field == "index" then
    local binder =
      terralib.newsymbol(c.legion_domain_point_t, pattern.binder)
    binders[pattern.binder] = binder
    actions = quote
      [actions]
      var [binder] = c.legion_task_get_index_point([task_var])
    end
  elseif pattern.field == "target" then
    local binder = terralib.newsymbol(c.legion_processor_t, pattern.binder)
    binders[pattern.binder] = binder
    actions = quote
      [actions]
      var [binder] = c.legion_task_get_target_proc([task_var])
    end
  else
    log.error(elem, "field " .. pattern.field ..
    " is not supported for pattern match")
  end
  return actions
end

function codegen.elem_pattern_match(binders, task_var, elem)
  local actions = quote end
  elem.patterns:map(function(pattern)
    actions = quote
      [actions]
      [codegen.pattern_match(binders, task_var, pattern)]
    end
  end)
  return actions
end

function codegen.select_task_options(rules, automata, signature,
                                     mapper_state_type)
  local rt_var = terralib.newsymbol(c.legion_mapper_runtime_t)
  local ctx_var = terralib.newsymbol(c.legion_mapper_context_t)
  local task_var = terralib.newsymbol(c.legion_task_t)
  local options_var = terralib.newsymbol(&c.legion_task_options_t)
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
    [last_elems:map(std.curry2(codegen.elem_pattern_match, binders, task_var))]
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
    [options_var].map_locally = false
    [options_var].stealable = false
    [body]
  end
end

function codegen.slice_task(rules, automata, state_id, signature, mapper_state_type)
  local rt_var = terralib.newsymbol(c.legion_mapper_runtime_t)
  local ctx_var = terralib.newsymbol(c.legion_mapper_context_t)
  local task_var = terralib.newsymbol(c.legion_task_t)
  local slice_task_output_var = terralib.newsymbol(c.legion_slice_task_output_t)
  local state_var = terralib.newsymbol(&mapper_state_type)
  local slice_cache_var = terralib.newsymbol(c.bishop_slice_cache_t)
  local point_var = terralib.newsymbol(c.legion_domain_point_t)
  local body = quote end

  local task_rules =
    data.filter(function(rule) return rule.rule_type == "task" end, rules)
  local last_elems =
    task_rules:map(function(rule)
      return rule.selector.elements[#rule.selector.elements]
    end)
  local binders = {}
  -- TODO: handle binder naming collision
  last_elems:map(function(elem)
    elem.patterns:map(function(pattern)
      if pattern.field == "index" then
        local binder =
          terralib.newsymbol(c.legion_domain_point_t, pattern.binder)
        binders[pattern.binder] = binder
        body = quote
          [body]
          var [binder] = [point_var]
        end
      end
    end)
  end)

  local task_properties = merge_task_properties(task_rules)
  local target = task_properties.target
  local value = codegen.expr(binders, state_var, target)
  -- TODO: distribute across slices processors in the list,
  --       instead of assigning all to the first processor
  if std.is_processor_list_type(target.expr_type) then
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
    var singleton : c.legion_domain_t
    var dim = [point_var].dim
    singleton.dim = dim
    singleton.is_id = 0
    for idx = 0, dim do
      var c = [point_var].point_data[idx]
      singleton.rect_data[idx] = c
      singleton.rect_data[idx + dim] = c
    end
    var slice = c.legion_task_slice_t {
      domain = singleton,
      proc = [value.value],
      recurse = false,
      stealable = false,
    }
    c.legion_slice_task_output_slices_add([slice_task_output_var], slice)
  end

  local selector_summary =
    terralib.constant(rawstring, tostring_selectors(task_rules))
  return terra(ptr : &opaque, [rt_var], [ctx_var], [task_var],
               slice_task_input : c.legion_slice_task_input_t,
               [slice_task_output_var])
    c.bishop_logger_debug("[slice_task] merged from %s", selector_summary)
    var [state_var] = [&mapper_state_type](ptr)
    var [slice_cache_var] = [state_var].[slice_field_name][ [state_id] ]

    if not c.bishop_slice_cache_has_cached_slices(
        [slice_cache_var], slice_task_input.domain) then
      var iterator = c.legion_domain_point_iterator_create(slice_task_input.domain)
      while c.legion_domain_point_iterator_has_next(iterator) do
        var [point_var] = c.legion_domain_point_iterator_next(iterator)
        [body]
      end
      -- TODO: set true when in debug mode
      c.legion_slice_task_output_verify_correctness_set([slice_task_output_var], false)
      c.legion_domain_point_iterator_destroy(iterator)
      c.bishop_slice_cache_add_entry([slice_cache_var], slice_task_input.domain,
          [slice_task_output_var])
    else
      c.bishop_slice_cache_copy_cached_slices([slice_cache_var],
          slice_task_input.domain, [slice_task_output_var])
    end
  end
end

function codegen.map_task(rules, automata, state_id, signature, mapper_state_type)
  local rt_var = terralib.newsymbol(c.legion_mapper_runtime_t)
  local ctx_var = terralib.newsymbol(c.legion_mapper_context_t)
  local task_var = terralib.newsymbol(c.legion_task_t)
  local map_task_input_var = terralib.newsymbol(c.legion_map_task_input_t)
  local map_task_output_var = terralib.newsymbol(c.legion_map_task_output_t)
  local state_var = terralib.newsymbol(&mapper_state_type)
  local layout_arr_var = terralib.newsymbol(&c.legion_layout_constraint_set_t)
  local instance_cache_var = terralib.newsymbol(c.bishop_instance_cache_t)
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
    [default_region_pattern_matches:map(std.curry2(codegen.pattern_match,
                                                   binders, task_var))]
    [last_elems:map(std.curry2(codegen.elem_pattern_match, binders, task_var))]
  end

  -- generate task mapping code
  local task_properties = merge_task_properties(task_rules)
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

  -- generate region mapping code
  body = quote
    [body]
    c.legion_map_task_output_chosen_instances_clear_all([map_task_output_var])
  end
  assert(signature)
  local rule_properties = merge_region_properties(region_rules, signature)
  for idx = 1, #rule_properties do
    local privilege =
      regent_std.privilege_mode(signature.reqs[idx].privilege)
    if privilege == c.NO_ACCESS then
      body = quote
        [body]
        do
          c.legion_map_task_output_chosen_instances_add(
            [map_task_output_var], [&c.legion_physical_instance_t](0), 0)
        end
      end
    else
      local req_var = terralib.newsymbol(c.legion_region_requirement_t)
      local fields_var =
        terralib.newsymbol(
            c.legion_field_id_t[#signature.reqs[idx].fields])
      local region_var = terralib.newsymbol(c.legion_logical_region_t)

      local target =
        codegen.expr(binders, state_var, rule_properties[idx].target)
      if std.is_memory_list_type(rule_properties[idx].target.expr_type) then
        local result = terralib.newsymbol(c.legion_memory_t)
        target.actions = quote
          [target.actions]
          var [result]
          if [target.value].size > 0 then
            [result] = [target.value].list[0]
          else
            [result] = c.bishop_get_no_memory()
          end
        end
        target.value = result
      end

      local level = rule_properties[idx].create.value
      if privilege == c.REDUCE or level == "demand" then
        local layout_var = terralib.newsymbol(c.legion_layout_constraint_set_t)
        local layout_init = quote
          [layout_var] = c.legion_layout_constraint_set_create()
          var fields_size = [#signature.reqs[idx].fields]
          var [fields_var]
          c.legion_region_requirement_get_privilege_fields([req_var],
            [fields_var], fields_size)
          c.legion_layout_constraint_set_add_field_constraint(
            [layout_var], [fields_var], fields_size, false, false)
        end
        if privilege == c.REDUCE then
          layout_init = quote
            [layout_init]
            var redop = c.legion_region_requirement_get_redop([req_var])
            c.legion_layout_constraint_set_add_specialized_constraint(
              [layout_var], c.REDUCTION_FOLD_SPECIALIZE, redop)
          end
        else
          layout_init = quote
            [layout_init]
            var dims : uint[4]
            dims[0], dims[1], dims[2], dims[3] =
              c.DIM_X, c.DIM_Y, c.DIM_Z, c.DIM_F
            c.legion_layout_constraint_set_add_ordering_constraint(
              [layout_var], dims, 4, false)
          end
        end

        layout_init = quote
          var [layout_var] = [layout_arr_var][ [idx - 1] ]

          -- TODO: invalidate cached layout constraints if necessary
          if [layout_var].impl == [&opaque](0) then
            [layout_init]
            c.legion_layout_constraint_set_add_memory_constraint(
              [layout_var], c.legion_memory_kind([target.value]))
            c.bishop_logger_debug(
              "[map_task] initialize layout constraints for region %d", [idx - 1])
            [layout_arr_var][ [idx - 1] ] = [layout_var]
          end
        end

        local inst_var = terralib.newsymbol(c.legion_physical_instance_t)
        local inst_creation = quote
          var success =
            c.legion_mapper_runtime_create_physical_instance_layout_constraint(
              [rt_var], [ctx_var], [target.value], [layout_var], &[region_var],
              1, &[inst_var], true, 0)
          std.assert(success, "instance creation must succeed")
        end

        body = quote
          [body]
          do
            [target.actions]
            var [inst_var]
            var [req_var] = c.legion_task_get_region([task_var], [idx - 1])
            var [region_var] = c.legion_region_requirement_get_region([req_var])
            [layout_init]
            [inst_creation]
            c.legion_map_task_output_chosen_instances_add([map_task_output_var],
              &[inst_var], 1)
            c.legion_physical_instance_destroy([inst_var])
          end
        end
      else
        assert(privilege ~= c.REDUCE and (level == "allow" or level == "forbid"))

        local inst_var = terralib.newsymbol(&c.legion_physical_instance_t)
        local layout_var = terralib.newsymbol(c.legion_layout_constraint_set_t)
        local cache_success_var = terralib.newsymbol(bool)
        local cache_init = quote
          --- TODO: Some layout constraints may require multiple instances
          [inst_var] = [&c.legion_physical_instance_t](
              c.malloc([terralib.sizeof(c.legion_physical_instance_t)]))
          var [layout_var] = c.legion_layout_constraint_set_create()
          var fields_size = [#signature.reqs[idx].fields]
          var [fields_var]
          c.legion_region_requirement_get_privilege_fields([req_var], [fields_var],
              fields_size)
          c.legion_layout_constraint_set_add_field_constraint([layout_var], [fields_var],
              fields_size, false, false)
          var dims : uint[4]
          dims[0], dims[1], dims[2], dims[3] = c.DIM_X, c.DIM_Y, c.DIM_Z, c.DIM_F
          c.legion_layout_constraint_set_add_ordering_constraint([layout_var], dims, 4, false)
        end

        if level == "forbid" then
          cache_init = quote
            [cache_init]
            var success =
              c.legion_mapper_runtime_find_physical_instance_layout_constraint(
                [rt_var], [ctx_var], [target.value], [layout_var], &[region_var],
                1, [inst_var], true, false)
            std.assert(success, "instance must be found")
          end
        else
          cache_init = quote
            [cache_init]
            var created : bool
            var success =
              c.legion_mapper_runtime_find_or_create_physical_instance_layout_constraint(
                [rt_var], [ctx_var], [target.value], [layout_var], &[region_var],
                1, [inst_var], &created, true, 0, false)
            std.assert(success, "instance creation must succeed")
          end
        end
        cache_init = quote
          [cache_init]
          [cache_success_var] =
            c.bishop_instance_cache_register_instances([instance_cache_var], [idx - 1],
              [region_var], [target.value], [inst_var])
        end

        body = quote
          [body]
          do
            [target.actions]
            var [req_var] = c.legion_task_get_region([task_var], [idx - 1])
            var [region_var] = c.legion_region_requirement_get_region([req_var])

            var [cache_success_var] = true
            var [inst_var] = c.bishop_instance_cache_get_cached_instances(
                [instance_cache_var], [idx - 1], [region_var], [target.value])
            if [inst_var] ~= [&c.legion_physical_instance_t](nil) then
              var success =
                c.legion_mapper_runtime_acquire_instances([rt_var], [ctx_var],
                    [inst_var], 1)
              std.assert(success, "instance acquire must succeed")
            else
              -- TODO: invalidate cached instances
              [cache_init]
              c.bishop_logger_debug(
                "[map_task] initialize instance cache for region %d", [idx - 1])
            end
            c.legion_map_task_output_chosen_instances_add([map_task_output_var],
              [inst_var], 1)
            if not[cache_success_var] then
              c.free([inst_var])
            end
          end
        end
      end
    end
  end

  local selector_summary =
    terralib.constant(rawstring, tostring_selectors(rules))
  local f = terra(ptr : &opaque, [rt_var], [ctx_var], [task_var],
               [map_task_input_var], [map_task_output_var])
    c.bishop_logger_debug("[map_task] merged from %s",
                          selector_summary)
    var [state_var] = [&mapper_state_type](ptr)
    var [layout_arr_var] = [state_var].[layout_field_name][ [state_id] ]
    var [instance_cache_var] = [state_var].[instance_field_name][ [state_id] ]
    [body]
  end
  return f
end

function codegen.mapper_init(assignments, automata, signatures)
  local entries = terralib.newlist()
  local binders = {}
  -- TODO: we might need to randomly generate this name with multiple mappers
  local mapper_state_type = terralib.types.newstruct("mapper_state")
  mapper_state_type.entries = assignments:map(function(assignment)
    assert(assignment.binder ~= layout_field_name)
    return {
      field = assignment.binder,
      type = codegen.type(assignment.value.expr_type),
    }
  end)
  mapper_state_type.entries:insertall({
    { field = layout_field_name,   type = &&c.legion_layout_constraint_set_t, },
    { field = instance_field_name, type = &c.bishop_instance_cache_t,         },
    { field = slice_field_name,    type = &c.bishop_slice_cache_t,            },
  })
  local mapper_state_var = terralib.newsymbol(&mapper_state_type)
  local mapper_init

  local max_state_id = 0
  for state, _ in pairs(automata.states) do
    if max_state_id < state.id then
      max_state_id = state.id
    end
  end
  max_state_id = max_state_id + 1
  local cache_init = quote
    [mapper_state_var].[layout_field_name] =
      [&&c.legion_layout_constraint_set_t](
        c.malloc([sizeof(&c.legion_layout_constraint_set_t)] * [max_state_id]))
    [mapper_state_var].[instance_field_name] =
      [&c.bishop_instance_cache_t](
        c.malloc([sizeof(c.bishop_instance_cache_t)] * [max_state_id]))
    [mapper_state_var].[slice_field_name] =
      [&c.bishop_slice_cache_t](
        c.malloc([sizeof(c.bishop_instance_cache_t)] * [max_state_id]))
  end
  for state, _ in pairs(automata.states) do
    if state ~= automata.initial then
      assert(state.last_task_symbol)
      local signature = signatures[state.last_task_symbol.task_name]
      cache_init = quote
        [cache_init]
        [mapper_state_var].[layout_field_name][ [state.id] ] =
          [&c.legion_layout_constraint_set_t](
            c.malloc([sizeof(c.legion_layout_constraint_set_t)] *
                     [#signature.reqs]))
        [mapper_state_var].[instance_field_name][ [state.id] ] =
          c.bishop_instance_cache_create()
        [mapper_state_var].[slice_field_name][ [state.id] ] =
          c.bishop_slice_cache_create()
        for idx = 0, [#signature.reqs] do
          [mapper_state_var].[layout_field_name][ [state.id] ][idx].impl =
            [&opaque](0)
        end
      end
    end
  end

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
    [cache_init]
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
  local state_to_mapper_impl = terralib.newlist()

  for state, _ in pairs(automata.states) do
    if state ~= automata.initial then
      -- every state now has a single task that it matches with
      assert(state.last_task_symbol)
      local selected_rules = terralib.newlist()
      for idx = 1, #rules do
        if state.tags[idx] then
          selected_rules:insert(rules[idx])
        end
      end
      local signature =
        signatures[state.last_task_symbol.task_name]
      local select_task_options =
        codegen.select_task_options(selected_rules, automata, signature,
                                    mapper_state_type)
      local slice_task =
        codegen.slice_task(selected_rules, automata, state.id, signature,
                             mapper_state_type)
      local map_task =
        codegen.map_task(selected_rules, automata, state.id, signature,
                         mapper_state_type)
      state_to_mapper_impl[state.id] = {
        select_task_options = select_task_options,
        slice_task = slice_task,
        map_task = map_task,
      }
    end
  end

  return state_to_mapper_impl
end

function codegen.automata(automata)
  local all_symbols = automata.all_symbols
  local state_to_transition_impl = terralib.newlist()
  for state, _ in pairs(automata.states) do
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
        local value = codegen.expr(binders, nil, symbol.constraint.value)
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
      terra([task_var]) : c.bishop_matching_state_t
        [body]
      end
  end
  return state_to_transition_impl
end

function codegen.mapper(node)
  local mapper_init, mapper_state_type =
    codegen.mapper_init(node.assignments, node.automata, node.task_signatures)

  local state_to_mapper_impl = codegen.rules(node.rules, node.automata,
                                             node.task_signatures,
                                             mapper_state_type)

  local state_to_transition_impl = codegen.automata(node.automata)

  -- install codegen hook for regent
  local transition_impls = terralib.newlist()
  for idx = 0, #state_to_transition_impl + 1 do
    transition_impls:insert(state_to_transition_impl[idx])
  end
  local transition_impl_tbl =
    terralib.constant(`arrayof([c.bishop_transition_fn_t], [transition_impls]))
  regent_codegen_hooks.set_update_mapping_tag(
    terra(task : c.legion_task_t)
      var tag = c.legion_task_get_tag(task)
      var prev_tag = tag
      while true do
        var fn : c.bishop_transition_fn_t =
          [transition_impl_tbl][prev_tag]
        tag = fn(task)
        if tag == prev_tag then break end
        prev_tag = tag
      end
      return tag
    end)

  return {
    mapper_init = mapper_init,
    state_to_transition_impl = state_to_transition_impl,
    state_to_mapper_impl = state_to_mapper_impl,
  }
end

return codegen
