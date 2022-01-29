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

-- Bishop AST

-- AST factory copied from Regent source
local ast_factory = {}

local function make_factory(name)
  return setmetatable(
    {
      parent = false,
      name = name,
      expected_fields = false,
      print_collapsed = false,
    },
    ast_factory)
end

local ast = make_factory("ast")
ast.make_factory = make_factory

-- Nodes

local ast_node = {}

function ast_node:__index(field)
  local value = ast_node[field]
  if value ~= nil then
    return value
  end
  local node_type = tostring(rawget(self, "node_type")) or "(unknown)"
  error(node_type .. " has no field '" .. field .. "' (in lookup)", 2)
end

function ast_node:__newindex(field, value)
  local node_type = tostring(rawget(self, "node_type")) or "(unknown)"
  error(node_type .. " has no field '" .. field .. "' (in assignment)", 2)
end

function ast.is_node(node)
  return type(node) == "table" and getmetatable(node) == ast_node
end

local function ast_node_tostring(node, indent, pretty)
  local newline = "\n"
  local spaces = string.rep("  ", indent)
  local spaces1 = string.rep("  ", indent + 1)
  if ast.is_node(node) then
    local collapsed = node.node_type.print_collapsed
    if collapsed then
      newline = ""
      spaces = ""
      spaces1 = ""
    end
    local str = tostring(node.node_type) .. "(" .. newline
    for k, v in pairs(node) do
      if k ~= "node_type" and k ~= "unparse" and
        (not pretty or k ~= "position") then
        str = str .. spaces1 .. k .. " = " ..
          ast_node_tostring(v, indent + 1, pretty) .. "," .. newline
      end
    end
    return str .. spaces .. ")"
  elseif terralib.islist(node) then
    local str = "{" .. newline
    for i, v in ipairs(node) do
      str = str .. spaces1 ..
        ast_node_tostring(v, indent + 1, pretty) .. "," .. newline
    end
    return str .. spaces .. "}"
  elseif type(node) == "string" then
    return string.format("%q", node)
  else
    return tostring(node)
  end
end

function ast_node:__tostring()
  return ast_node_tostring(self, 0)
end

function ast_node:printpretty()
  print(ast_node_tostring(self, 0, true))
end

function ast_node:unparse()
  return rawget(self, "unparse")()
end

function ast_node:is(node_type)
  return self.node_type:is(node_type)
end

function ast_node:type()
  return self.node_type
end

function ast_node:__call(fields_to_update)
  local ctor = rawget(self, "node_type")
  local values = {}
  for _, f in ipairs(ctor.expected_fields) do
    values[f] = self[f]
  end
  for f, v in pairs(fields_to_update) do
    if f ~= "unparse" and f ~= "node_type" then
      if values[f] == nil then
        error(tostring(ctor) .. " does not require argument '" .. f .. "'", 2)
      end
      values[f] = v
    end
  end
  return ctor(values)
end

-- Constructors

local ast_ctor = {}

function ast_ctor:__index(field)
  local value = ast_ctor[field]
  if value ~= nil then
    return value
  end
  error(tostring(self) .. " has no field '" .. field .. "'", 2)
end

function ast_ctor:__call(node)
  assert(type(node) == "table", tostring(self) .. " expected table")
  for i, f in ipairs(self.expected_fields) do
    if rawget(node, f) == nil then
      error(tostring(self) .. " missing required argument '" .. f .. "'", 2)
    end
  end
  rawset(node, "node_type", self)
  if rawget(self, "unparse") then rawset(node, "unparse", self.unparse) end
  setmetatable(node, ast_node)
  return node
end

function ast_ctor:__tostring()
  return tostring(self.parent) .. "." .. self.name
end

function ast_ctor:is(node_type)
  return self == node_type or self.parent:is(node_type)
end

-- Factories

local function merge_fields(...)
  local keys = {}
  local result = terralib.newlist({})
  for _, fields in ipairs({...}) do
    if fields then
      for _, field in ipairs(fields) do
        if keys[field] then
          error("multiple definitions of field " .. field)
        end
        keys[field] = true
        result:insert(field)
      end
    end
  end
  return result
end

function ast_factory:__index(field)
  local value = ast_factory[field]
  if value ~= nil then
    return value
  end
  error(tostring(self) .. " has no field '" .. field .. "'", 2)
end

function ast_factory:inner(ctor_name, expected_fields, print_collapsed)
  local ctor = setmetatable(
    {
      parent = self,
      name = ctor_name,
      expected_fields = merge_fields(self.expected_fields, expected_fields),
      print_collapsed = (print_collapsed == nil and self.print_collapsed) or print_collapsed or false
    }, ast_factory)

  assert(rawget(self, ctor_name) == nil,
         "multiple definitions of constructor " .. ctor_name)
  self[ctor_name] = ctor
  return ctor
end

function ast_factory:leaf(ctor_name, expected_fields, print_collapsed)
  local ctor = setmetatable(
    {
      parent = self,
      name = ctor_name,
      expected_fields = merge_fields(self.expected_fields, expected_fields),
      print_collapsed = (print_collapsed == nil and self.print_collapsed) or print_collapsed or false
    }, ast_ctor)

  assert(rawget(self, ctor_name) == nil,
         "multiple definitions of constructor " .. ctor_name)
  self[ctor_name] = ctor
  return ctor
end

function ast_factory:is(node_type)
  return self == node_type or (self.parent and self.parent:is(node_type))
end

function ast_factory:__tostring()
  if self.parent then
    return tostring(self.parent) .. "." .. self.name
  end
  return self.name
end

-- Location

ast:leaf("Position", { "filename", "linenumber", "offset" }, true)

function ast.save(p)
  local token = p:cur()
  return ast.Position {
    filename = p.source,
    linenumber = token.linenumber,
    offset = token.offset,
  }
end

function ast.trivial_pos()
  return ast.Position {
    filename = "",
    linenumber = 0,
    offset = 0,
  }
end

-- Node Types (Unspecialized)

local function unparse_all(list)
  return list:map(function(e) return e:unparse() end)
end

ast:inner("untyped", { "position" })

ast.untyped:leaf("Mapper", { "rules", "assignments" })
function ast.untyped.Mapper:unparse()
  local str = ""
  for i = 1, #self.assignments do
    str = str .. self.assignments[i]:unparse() .. "\n"
  end
  for i = 1, #self.rules do
    str = str .. self.rules[i]:unparse()
  end
  return str
end
ast.untyped:leaf("Rule", { "selectors", "properties" })
function ast.untyped.Rule:unparse()
  local str = self.selectors[1]:unparse()
  for i = 2, #self.selectors do
    str = str .. ", " .. self.selectors[i]:unparse()
  end
  str = str .. " {\n"
  for i = 1, #self.properties do
    str = str .. "  " .. self.properties[i]:unparse() .. "\n"
  end
  str = str .. "}\n"
  return str
end

ast.untyped:leaf("Selector", { "elements" })
function ast.untyped.Selector:unparse()
  local str = self.elements[1]:unparse()
  for i = 2, #self.elements do
    str = str .. " " .. self.elements[i]:unparse()
  end
  return str
end

ast.untyped:inner("element", { "name", "classes", "constraints", "patterns" })
function ast.untyped.element:unparse()
  local str = ""
  for i = 1, #self.name do
    str = str .. "#" .. self.name[i]
  end
  for i = 1, #self.classes do
    str = str .. "." .. self.classes[i]
  end
  if #self.constraints > 0 then
    local const_str = table.concat(unparse_all(self.constraints), "")
    str = str .. const_str
  end
  if #self.patterns > 0 then
    local pat_str = table.concat(unparse_all(self.patterns), "")
    str = str .. pat_str
  end
  return str
end

ast.untyped.element:leaf("Task", {})
function ast.untyped.element.Task:unparse()
  local str = "task"
  return str .. ast.untyped.element.unparse(self)
end
ast.untyped.element:leaf("Region", {})
function ast.untyped.element.Region:unparse()
  local str = "region"
  return str .. ast.untyped.element.unparse(self)
end

ast.untyped:leaf("Property", { "field", "value" })
function ast.untyped.Property:unparse()
  return self.field .. " : " .. self.value:unparse() .. ";"
end

ast.untyped:leaf("Constraint", { "field", "value" })
function ast.untyped.Constraint:unparse()
  return "[" .. self.field .. "=" .. self.value:unparse() .. "]"
end

ast.untyped:leaf("FilterConstraint", { "field", "value" })
function ast.untyped.FilterConstraint:unparse()
  return "[" ..self.field .. "=" .. self.value:unparse() .. "]"
end

ast.untyped:leaf("PatternMatch", { "field", "binder" })
function ast.untyped.PatternMatch:unparse()
  return "[" .. self.field .. "=$" .. self.binder .. "]"
end

ast.untyped:leaf("Assignment", { "binder", "value" })
function ast.untyped.Assignment:unparse()
  return "$" .. self.binder .. " = " .. self.value:unparse()
end

ast.untyped:inner("expr")
ast.untyped.expr:leaf("Unary", { "rhs", "op" })
function ast.untyped.expr.Unary:unparse()
  return tostring(self.op) .. self.rhs:unparse()
end

ast.untyped.expr:leaf("Binary", { "lhs", "rhs", "op" })
function ast.untyped.expr.Binary:unparse()
  return self.lhs:unparse() .. tostring(self.op) .. self.rhs:unparse()
end

ast.untyped.expr:leaf("Ternary", { "cond", "true_expr", "false_expr" })
function ast.untyped.expr.Ternary:unparse()
  return self.cond:unparse() .. " ? " ..
         self.true_expr:unparse() .. " : " ..
         self.false_expr:unparse()
end

ast.untyped.expr:leaf("Index", { "value", "index" })
function ast.untyped.expr.Index:unparse()
  return self.value:unparse() .. "[" .. self.index:unparse() .. "]"
end

ast.untyped.expr:leaf("Filter", { "value", "constraints" })
function ast.untyped.expr.Filter:unparse()
  local const_str = table.concat(unparse_all(self.constraints), " and ")
  return self.value:unparse() .. "[" .. const_str .. "]"
end

ast.untyped.expr:leaf("Field", { "value", "field" })
function ast.untyped.expr.Field:unparse()
  return self.value:unparse() .. "." .. self.field
end

ast.untyped.expr:leaf("Constant", { "value" })
function ast.untyped.expr.Constant:unparse()
  return tostring(self.value)
end
ast.untyped.expr:leaf("Variable", { "value" })
function ast.untyped.expr.Variable:unparse()
  return self.value
end
ast.untyped.expr:leaf("Keyword", { "value" })
function ast.untyped.expr.Keyword:unparse()
  return self.value
end

ast:inner("typed", { "position" })

ast.typed:leaf("Mapper", { "rules", "assignments" })
ast.typed:leaf("Rule", { "rule_type", "selector", "properties" })
function ast.typed.Rule:unparse()
  local str = self.selector:unparse()
  str = str .. " {\n"
  for i = 1, #self.properties do
    str = str .. "  " .. self.properties[i]:unparse() .. "\n"
  end
  str = str .. "}\n"
  return str
end

ast.typed:leaf("Selector", { "type", "elements", })
function ast.typed.Selector:unparse()
  return table.concat(unparse_all(self.elements), " ")
end

ast.typed:inner("element", { "name", "classes", "constraints", "patterns" })
ast.typed.element.unparse = ast.untyped.element.unparse

ast.typed.element:leaf("Task", {})
function ast.typed.element.Task:unparse()
  return "task" .. ast.typed.element.unparse(self)
end
ast.typed.element:leaf("Region", {})
function ast.typed.element.Region:unparse()
  return "region" .. ast.typed.element.unparse(self)
end

ast.typed:leaf("Property", { "field", "value" })
ast.typed.Property.unparse = ast.untyped.Property.unparse
ast.typed:leaf("Constraint", { "field", "value" })
ast.typed.Constraint.unparse = ast.untyped.Constraint.unparse
ast.typed:leaf("FilterConstraint", { "field", "value" })
ast.typed.FilterConstraint.unparse = ast.untyped.FilterConstraint.unparse
ast.typed:leaf("PatternMatch", { "field", "binder" })
ast.typed.PatternMatch.unparse = ast.untyped.PatternMatch.unparse
ast.typed:leaf("Assignment", { "binder", "value" })
function ast.typed.Assignment:unparse()
  return "$" .. self.binder .. " = " .. self.value:unparse()
end

ast.typed:inner("expr", { "expr_type" })
ast.typed.expr:leaf("Unary", { "rhs", "op" })
function ast.typed.expr.Unary:unparse()
  return tostring(self.op) .. self.rhs:unparse()
end
ast.typed.expr:leaf("Binary", { "lhs", "rhs", "op" })
function ast.typed.expr.Binary:unparse()
  return self.lhs:unparse() .. tostring(self.op) .. self.rhs:unparse()
end
ast.typed.expr:leaf("Ternary", { "cond", "true_expr", "false_expr" })
function ast.typed.expr.Ternary:unparse()
  return self.cond:unparse() .. " ? " ..
         self.true_expr:unparse() .. " : " ..
         self.false_expr:unparse()
end
ast.typed.expr:leaf("Index", { "value", "index" })
function ast.typed.expr.Index:unparse()
  return self.value:unparse() .. "[" .. self.index:unparse() .. "]"
end
ast.typed.expr:leaf("Filter", { "value", "constraints" })
function ast.typed.expr.Filter:unparse()
  local const_str = table.concat(unparse_all(self.constraints), " and ")
  return self.value:unparse() .. "[" .. const_str .. "]"
end
ast.typed.expr:leaf("Field", { "value", "field" })
function ast.typed.expr.Field:unparse()
  return self.value:unparse() .. "." .. self.field
end
ast.typed.expr:leaf("Coerce", { "value" })
ast.typed.expr:leaf("Constant", { "value" })
ast.typed.expr.Constant.unparse = ast.untyped.expr.Constant.unparse
ast.typed.expr:leaf("Variable", { "value" })
ast.typed.expr.Variable.unparse = ast.untyped.expr.Variable.unparse
ast.typed.expr:leaf("Keyword", { "value" })
ast.typed.expr.Keyword.unparse = ast.untyped.expr.Keyword.unparse

ast:inner("optimized")

ast.optimized:leaf("Mapper", { "automata", "rules", "assignments",
                               "task_signatures" })

return ast
