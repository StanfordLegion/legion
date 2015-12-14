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


-- Bishop Specialization Pass

local ast = require("bishop/ast")
local log = require("bishop/log")
local std = require("bishop/std")

local function any(values)
  local tbl = {}
  for _, value in pairs(values) do tbl[value] = true end
  return function(value)
    return tbl[value] ~= nil
  end
end

local function number(value)
  return type(value) == "number"
end

local function integer(value)
  return number(value) and math.floor(value) == value
end

local function always(value)
  return true
end

local function either(f1, f2)
  return function(value)
    return f1(value) or f2(value)
  end
end

local compile_levels = any { "forbid", "allow", "demand" }

local property_checks = {
  block = {
    type = any { "region" },
    condition = always,
  },
  colocate = {
    type = any { "task" },
    condition = always,
  },
  composite = {
    type = any { "region" },
    condition = any { "forbid", "demand" },
  },
  create = {
    type = any { "region" },
    condition = compile_levels,
  },
  generate = {
    type = any { "task", "for", "while", "do" },
    condition = any { "aot", "jit" },
  },
  index_launch = {
    type = any { "task", "for", "while", "do" },
    condition = compile_levels,
  },
  inline = {
    type = any { "task", "for", "while", "do" },
    condition = compile_levels,
  },
  isa = {
    type = any { "task" },
    condition = any { "x86", "cuda", "arm", "ptx",
                   "lua", "terra", "llvm", "gl",
                   "sse", "sse2", "sse3", "sse4",
                   "avx", "avx2", "fma", "mic",
                   "sm10", "sm20", "sm30", "sm35",
                   "neon" }
  },
  spmd = {
    type = any { "task", "for", "while", "do" },
    condition = compile_levels,
  },
  spmd_subranks = {
    type = any { "task", "for", "while", "do" },
    condition = integer,
  },
  target = {
    type = any { "task", "region" },
    condition = always,
  },
  unroll = {
    type = any { "task", "for", "while", "do" },
    condition = compile_levels,
  },
  unroll_factor = {
    type = any { "task", "for", "while", "do" },
    condition = integer,
  },
  vectorize = {
    type = any { "task", "for", "while", "do" },
    condition = compile_levels,
  },
}

local pattern_match_checks = {
  task = any { "index", "target", "isa" },
  region = any { "target", "instances", "requested", "declared" },
}

local specialize = {}

function specialize.expr(node)
  if node:is(ast.unspecialized.expr.Index) then
    local value = specialize.expr(node.value)
    local index = specialize.expr(node.index)
    return ast.specialized.expr.Index {
      value = value,
      index = index,
      position = node.position,
    }
  elseif node:is(ast.unspecialized.expr.Filter) then
    local value = specialize.expr(node.value)
    local constraints = node.constraints:map(function(constraint)
      local field = constraint.field
      local value
      if constraint:is(ast.unspecialized.Constraint) then
        value = constraint.value
      else assert(constraint:is(ast.unspecialized.PatternMatch)) 
        value = ast.unspecialized.expr.Variable {
          value = constraint.binder,
          position = constraint.position,
        }
      end
      return ast.specialized.FilterConstraint {
        field = field,
        value = specialize.expr(value),
        position = constraint.position,
      }
    end)
    local new_node = ast.specialized.expr.Filter {
      value = value,
      constraints = constraints,
      position = node.position,
    }
    return new_node
  elseif node:is(ast.unspecialized.expr.Field) then
    local value = specialize.expr(node.value)
    return ast.specialized.expr.Field {
      value = value,
      field = node.field,
      position = node.position,
    }
  elseif node:is(ast.unspecialized.expr.Constant) then
    return ast.specialized.expr.Constant {
      value = node.value,
      position = node.position,
    }
  elseif node:is(ast.unspecialized.expr.Keyword) then
    return ast.specialized.expr.Keyword {
      value = node.value,
      position = node.position,
    }
  elseif node:is(ast.unspecialized.expr.Variable) then
    return ast.specialized.expr.Variable {
      value = node.value,
      position = node.position,
    }
  elseif node:is(ast.unspecialized.expr.Unary) then
    local rhs = specialize.expr(node.rhs)
    return ast.specialized.expr.Unary {
      op = node.op,
      rhs = rhs,
      position = node.position,
    }
  elseif node:is(ast.unspecialized.expr.Binary) then
    local lhs = specialize.expr(node.lhs)
    local rhs = specialize.expr(node.rhs)
    return ast.specialized.expr.Binary {
      op = node.op,
      lhs = lhs,
      rhs = rhs,
      position = node.position,
    }
  else
    assert(false, "unexpected node type: " .. tostring(node.node_type))
  end
end

function specialize.property(node, type)
  local field = node.field
  local value = specialize.expr(node.value)

  if not (property_checks[field] and
    property_checks[field].type(type)) then
    log.error(node, "unexpected property '" .. field ..
    "' for " .. type .. " rule")
  end
  if value:is(ast.specialized.expr.Keyword) then
    local keyword = value.value
    if not property_checks[field].condition(keyword) then
      log.error(node, "unexpected keyword '" .. keyword ..
      "' for property '" .. field .. "'")
    end
  end

  return ast.specialized.Property {
    field = field,
    value = value,
    position = node.position,
  }
end

function specialize.properties(list, type)
  return list:map(function(property)
    return specialize.property(property, type)
  end)
end

function specialize.element(node)
  local name = node.name
  local classes = node.classes
  local constraints = terralib.newlist()
  local patterns = terralib.newlist()
  local pattern_match_check
  if node:is(ast.unspecialized.element.Task) then
    pattern_match_check = pattern_match_checks.task
  else assert(node:is(ast.unspecialized.element.Region))
    pattern_match_check = pattern_match_checks.region
  end

  node.constraints:map(function(constraint)
    local binder = std.newsymbol()
    if pattern_match_check(constraint.field) and
      not property_checks[constraint.field] then
      log.error(constraint, "invalid constraint on field '" ..
        constraint.field .. "'")
    end
    patterns:insert(ast.specialized.PatternMatch {
      binder = binder,
      field = constraint.field,
      position = constraint.position,
    })
    constraints:insert(ast.specialized.Constraint {
      lhs = ast.specialized.expr.Variable {
        value = binder,
        position = constraint.position,
      },
      rhs = specialize.expr(constraint.value),
      position = constraint.position,
    })
  end)
  node.patterns:map(function(pattern)
    if not pattern_match_check(pattern.field) then
      log.error(pattern, "invalid pattern match on field '" ..
        pattern.field .. "'")
    end
    local variable = pattern.binder
    patterns:insert(ast.specialized.PatternMatch {
      binder = variable,
      field = pattern.field,
      position = pattern.position,
    })
  end)

  local ctor
  if node:is(ast.unspecialized.element.Task) then
    ctor = ast.specialized.element.Task
  else assert(node:is(ast.unspecialized.element.Region))
    ctor = ast.specialized.element.Region
  end

  return ctor {
    name = name,
    classes = classes,
    patterns = patterns,
    position = node.position,
  }, constraints
end

function specialize.selector(node)
  local elements = terralib.newlist()
  local constraints = terralib.newlist()

  for i = #node.elements, 1, -1 do
    local e, c = specialize.element(node.elements[i])
    elements:insert(e)
    constraints:insertall(c)
  end

  local type
  if elements[1]:is(ast.specialized.element.Task) then
    type = "task"
    if #elements > 1 then
      log.error(node,
        "task selectors with multiple elements are not supported yet")
    end
  else assert(elements[1]:is(ast.specialized.element.Region))
    type = "region"
    if not elements[2]:is(ast.specialized.element.Task) then
      log.error(elements[2],
        "region element should be preceded by task element in selectors")
    end
    if #elements > 2 then
      log.error(node,
        "region selectors with multiple task elements are not supported yet")
    end
  end

  return ast.specialized.Selector {
    type = type,
    elements = elements,
    constraints = constraints,
    position = node.position,
  }
end

function specialize.rule(node)
  local selector = specialize.selector(node.selectors)
  local properties = specialize.properties(node.properties, selector.type)

  local ctor
  if selector.type == "task" then
    ctor = ast.specialized.rule.Task
  else assert(selector.type == "region")
    ctor = ast.specialized.rule.Region
  end

  return ctor {
    selector = selector,
    properties = properties,
    position = node.position
  }
end

local function calculate_specificity(selector)
  local s = { 0, 0, 0, 0 }
  for i = 1, #selector.elements do
    local e = selector.elements[i]
    if #e.name > 0 then s[1] = s[1] + 1 end
    if #e.classes > 0 then s[2] = s[2] + 1 end
    if #e.patterns > 0 then s[3] = s[3] + 1 end
  end
  s[4] = #selector.elements
  return s
end

local function compare_rules(rule1, rule2)
  assert(rule2:is(rule1.node_type))
  local specificity1 = calculate_specificity(rule1.selector)
  local specificity2 = calculate_specificity(rule2.selector)
  for i = 1, 4 do
    if specificity1[i] >= specificity2[i] then return false end
  end
  return true
end

function specialize.rules(node)
  local flattened = terralib.newlist()
  node.rules:map(function(rule)
    rule.selectors:map(function(selector)
      flattened:insert(rule { selectors = selector })
    end)
  end)

  local rules = flattened:map(specialize.rule)
  local task_rules = terralib.newlist()
  local region_rules = terralib.newlist()

  rules:map(function(rule)
    if rule:is(ast.specialized.rule.Task) then
      task_rules:insert(rule)
    else assert(rule:is(ast.specialized.rule.Region))
      region_rules:insert(rule)
    end
  end)

  table.sort(task_rules, compare_rules)
  table.sort(region_rules, compare_rules)

  return ast.specialized.Rules {
    task_rules = task_rules,
    region_rules = region_rules,
    position = node.position,
  }
end

return specialize
