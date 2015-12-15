-- Copyright 2015 Stanford University, NVIDIA Corporation
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


-- Bishop Type Checking

local ast = require("bishop/ast")
local log = require("bishop/log")
local std = require("bishop/std")
local symbol_table = require("regent/symbol_table")

local type_check = {}

local function curry(f, a) return function(...) return f(a, ...) end end
local function curry2(f, a, b) return function(...) return f(a, b, ...) end end

local element_type_assignment = {
  task = {
    index = std.point_type,
    isa = std.isa_type,
    target = std.processor_type,
  },
  region = {
    -- instances = std.instance_list_type,
    -- requested = std.field_list_type,
    -- declared = std.field_list_type,
  },
}

local property_type_assignment = {
  task = {
    inline = { std.compile_option_type },
    isa = { std.isa_type },
    target = { std.processor_type, std.processor_list_type },
    vectorize = { std.compile_option_type },
  },
  region = {
    composite = { std.compile_option_type },
    create = { std.compile_option_type },
    target = { std.memory_type, std.memory_list_type },
  },
}

local keyword_type_assignment = {}
function keyword_type_assignment:assign_type(keywords, type)
  for _, keyword in ipairs(keywords) do self[keyword] = type end
end
keyword_type_assignment:assign_type({ "x86", "cuda", "arm", "ptx", "lua",
                                      "terra", "llvm", "gl", "sse", "sse2",
                                      "sse3", "sse4", "avx", "avx2", "fma",
                                      "mic", "sm10", "sm20", "sm30", "sm35",
                                      "neon" }, std.isa_type)
keyword_type_assignment:assign_type({ "global", "sysmem", "regmem", "fbmem",
                                      "zcmem", "disk", "hdf", "file",
                                      "l1cache", "l2cache", "l3cache" },
                                      std.memory_kind_type)

keyword_type_assignment:assign_type({ "forbid", "allow", "demand" },
                                    std.compile_option_type)
keyword_type_assignment:assign_type({ "processors" }, std.processor_list_type)
keyword_type_assignment:assign_type({ "memories" }, std.memory_list_type)

function type_check.filter_constraint(value_type, type_env, constraint)
  assert(constraint:is(ast.specialized.FilterConstraint))
  local value = type_check.expr(type_env, constraint.value)
  local invalid_field_msg = "invalid filter constraint on field '" ..
    constraint.field .. "' for " .. tostring(value_type)
  local type_error_msg = "expression of type '" .. tostring(value.expr_type) ..
    "' is invalid for filtering on field '" .. constraint.field .. "'"

  if std.is_processor_list_type(value_type) then
    if constraint.field == "isa" then
      if not std.is_isa_type(value.expr_type) then
        log.error(constraint.value, type_error_msg)
      end
    else
      log.error(constraint, invalid_field_msg)
    end

  elseif std.is_memory_list_type(value_type) then
    if constraint.field == "kind" then
      if not std.is_memory_kind_type(value.expr_type) then
        log.error(constraint.value, type_error_msg)
      end
    else
      log.error(constraint, invalid_field_msg)
    end

  else
    assert(false, "unreachable")
  end

  return ast.typed.FilterConstraint {
    field = constraint.field,
    value = value,
    position = constraint.position,
  }
end

function type_check.expr(type_env, expr)
  if expr:is(ast.specialized.expr.Unary) then
    local rhs = type_check.expr(type_env, expr.rhs)
    if expr.op == "-" then
      if std.is_point_type(rhs.expr_type) then
        rhs = ast.typed.expr.Coerce {
          value = rhs,
          expr_type = int,
          position = expr.position,
        }
      elseif rhs.expr_type ~= int then
        log.error(expr.rhs, "unary op " .. expr.op ..
          " requires the rhs to be of integer type")
      end
    else
      log.error(expr, "unexpected unary operation")
    end
    return ast.typed.expr.Unary {
      rhs = rhs,
      op = expr.op,
      expr_type = rhs.expr_type,
      position = expr.position,
    }

  elseif expr:is(ast.specialized.expr.Binary) then
    local lhs = type_check.expr(type_env, expr.lhs)
    local rhs = type_check.expr(type_env, expr.rhs)
    if not (expr.op == "+" or expr.op == "-" or
            expr.op == "*" or expr.op == "/" or expr.op == "%") then
      log.error(expr, "unexpected binary operation")
    end
    if std.is_point_type(lhs.expr_type) then
      lhs = ast.typed.expr.Coerce {
        value = lhs,
        expr_type = int,
        position = expr.position,
      }
    elseif lhs.expr_type ~= int then
      log.error(expr.lhs, "binary op " .. expr.op ..
        " requires the lhs to be of integer type")
    end

    if std.is_point_type(rhs.expr_type) then
      rhs = ast.typed.expr.Coerce {
        value = rhs,
        expr_type = int,
        position = expr.position,
      }
    elseif rhs.expr_type ~= int then
      log.error(expr.rhs, "binary op " .. expr.op ..
        " requires the rhs to be of integer type")
    end

    return ast.typed.expr.Binary {
      lhs = lhs,
      rhs = rhs,
      op = expr.op,
      expr_type = rhs.expr_type,
      position = expr.position,
    }

  elseif expr:is(ast.specialized.expr.Index) then
    local value = type_check.expr(type_env, expr.value)
    local index = type_check.expr(type_env, expr.index)
    if std.is_point_type(index.expr_type) then
      index = ast.typed.expr.Coerce {
        value = index,
        expr_type = int,
        position = expr.position,
      }
    elseif index.expr_type ~= int then
      log.error(expr.index, "indexing expression requires to have integer type," ..
        " but received '" .. tostring(index.expr_type) .. "'")
    end
    if not (std.is_list_type(value.expr_type) or
            std.is_point_type(value.expr_type)) then
      log.error(expr.value, "index access requires list or point type," ..
        " but received '" .. tostring(value.expr_type) .. "'")
    end
    local expr_type
    if std.is_point_type(value.expr_type) then
      expr_type = int
    elseif std.is_processor_list_type(value.expr_type) then
      expr_type = std.processor_type
    elseif std.is_memory_list_type(value.expr_type) then
      expr_type = std.memory_type
    else
      assert(false, "unreachable")
    end
    return ast.typed.expr.Index {
      value = value,
      index = index,
      expr_type = expr_type,
      position = expr.position,
    }

  elseif expr:is(ast.specialized.expr.Filter) then
    local value = type_check.expr(type_env, expr.value)
    if not std.is_list_type(value.expr_type) then
      log.error(expr.value, "filter is not valid on expressions of " ..
        "type '" .. tostring(value.expr_type) .. "'")
    end
    local constraints = expr.constraints:map(
      curry2(type_check.filter_constraint, value.expr_type, type_env))
    return ast.typed.expr.Filter {
      value = value,
      constraints = constraints,
      expr_type = value.expr_type,
      position = expr.position,
    }

  elseif expr:is(ast.specialized.expr.Field) then
    local value = type_check.expr(type_env, expr.value)
    if expr.field == "memories" then
      if not std.is_processor_type(value.expr_type) then
        log.error(expr, "value of type '" .. tostring(value.expr_type) ..
          "' does not have field '" .. expr.field .. "'")
      end
      local assigned_type = std.memory_list_type

      return ast.typed.expr.Field {
        value = value,
        field = expr.field,
        expr_type = assigned_type,
        position = expr.position,
      }

    elseif expr.field == "size" then
      if not std.is_list_type(value.expr_type) then
        log.error(expr, "value of type '" .. tostring(value.expr_type) ..
          "' does not have field '" .. expr.field .. "'")
      end
      local assigned_type = int

      return ast.typed.expr.Field {
        value = value,
        field = expr.field,
        expr_type = assigned_type,
        position = expr.position,
      }

    else
      log.error(expr, "unknown field access on field '" .. expr.field .. "'")
    end

  elseif expr:is(ast.specialized.expr.Constant) then
    local assigned_type = int
    if math.floor(expr.value) ~= expr.value then
      log.error(expr, "constant must be an integer value")
    end
    return ast.typed.expr.Constant {
      value = expr.value,
      expr_type = assigned_type,
      position = expr.position,
    }

  elseif expr:is(ast.specialized.expr.Variable) then
    local assigned_type = type_env:safe_lookup(expr.value)
    if not assigned_type then
      log.error(expr, "variable '$" .. expr.value ..
        "' was used without being bound by a pattern matching")
    end
    return ast.typed.expr.Variable {
      value = expr.value,
      expr_type = assigned_type,
      position = expr.position,
    }

  elseif expr:is(ast.specialized.expr.Keyword) then
    local assigned_type = keyword_type_assignment[expr.value]
    if not assigned_type then
      log.error(expr, "unknown keyword '" .. expr.value .. "'")
    end
    return ast.typed.expr.Keyword {
      value = expr.value,
      expr_type = assigned_type,
      position = expr.position,
    }

  else
    assert(false, "unexpected expression type")
  end
end

function type_check.element(type_env, element)
  local element_type
  local type_assignment
  local ctor
  if element:is(ast.specialized.element.Task) then
    element_type = "task"
    type_assignment = element_type_assignment.task
    ctor = ast.typed.element.Task
  elseif element:is(ast.specialized.element.Region) then
    element_type = "region"
    type_assignment = element_type_assignment.region
    ctor = ast.typed.element.Region
  else
    assert(false, "unexpected element type")
  end
  local patterns = element.patterns:map(function(pattern)
    local desired_type = type_assignment[pattern.field]
    local existing_type = type_env:safe_lookup(pattern.binder) or desired_type
    assert(desired_type)
    if desired_type ~= existing_type then
      log.error(pattern, "variable '$" .. pattern.binder .. "' was assigned" ..
        " to two different types '" .. tostring(existing_type) ..
        "' and '" .. tostring(desired_type) .. "'")
    end
    type_env:insert(element, pattern.binder, desired_type)
    return ast.typed.PatternMatch(pattern)
  end)
  return ctor {
    name = element.name,
    classes = element.classes,
    patterns = patterns,
    position = element.position,
  }
end

function type_check.constraint(type_env, constraint)
  local lhs = type_check.expr(type_env, constraint.lhs)
  local rhs = type_check.expr(type_env, constraint.rhs)
  if lhs.expr_type ~= rhs.expr_type then
    log.error(constraint, "type mismatch: '" .. tostring(lhs.expr_type) ..
      "' and '" .. tostring(rhs.expr_type) .. "'")
  end
  return ast.typed.Constraint {
    lhs = lhs,
    rhs = rhs,
    position = constraint.position,
  }
end

function type_check.property(rule_type, type_env, property)
  local value = type_check.expr(type_env, property.value)
  local desired_types = property_type_assignment[rule_type][property.field]
  assert(desired_types)
  local found = false
  for i = 1, #desired_types do
    if desired_types[i] == value.expr_type then
      found = true
      break
    end
  end
  if not found then
    log.error(property, "property '" .. property.field .. "' of " ..
      rule_type .. " rule cannot get assigned by an expression of type '" ..
      tostring(value.expr_type) .. "'")
  end

  return ast.typed.Property {
    field = property.field,
    value = value,
    position = property.position,
  }
end

function type_check.selector(type_env, selector)
  local elements = selector.elements:map(curry(type_check.element, type_env))
  local constraints =
    selector.constraints:map(curry(type_check.constraint, type_env))
  return ast.typed.Selector {
    type = selector.type,
    elements = elements,
    constraints = constraints,
    position = selector.position,
  }
end

function type_check.rule(rule_type, type_env, rule)
  local selector = type_check.selector(type_env, rule.selector)
  local properties =
    rule.properties:map(curry2(type_check.property, rule_type, type_env))
  return ast.typed.rule.Task {
    selector = selector,
    properties = properties,
    position = rule.position,
  }
end

function type_check.task_rule(type_env, rule)
  return type_check.rule("task", type_env, rule)
end
function type_check.region_rule(type_env, rule)
  return type_check.rule("region", type_env, rule)
end

function type_check.assignment(type_env, assignment)
  local value = type_check.expr(type_env, assignment.value)
  if type_env:safe_lookup(assignment.binder) then
    log.error(assignment, "variable '$" .. assignment.binder ..
      "' has been already defined")
  end
  type_env:insert(assignment, assignment.binder, value.expr_type)
  return ast.typed.Assignment {
    binder = assignment.binder,
    value = value,
    position = assignment.position,
  }
end

function type_check.mapper(mapper)
  local type_env = symbol_table:new_global_scope()
  local assignments = mapper.assignments:map(function(assignment)
    return type_check.assignment(type_env, assignment)
  end)

  local task_rules = mapper.task_rules:map(function(rule)
    local local_type_env = type_env:new_local_scope()
    return type_check.task_rule(local_type_env, rule)
  end)
  local region_rules = mapper.region_rules:map(function(rule)
    local local_type_env = type_env:new_local_scope()
    return type_check.region_rule(local_type_env, rule)
  end)
  return ast.typed.Mapper {
    assignments = assignments,
    task_rules = task_rules,
    region_rules = region_rules,
    position = mapper.position,
  }
end

return type_check
