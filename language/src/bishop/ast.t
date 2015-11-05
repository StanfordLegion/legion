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

local function ast_node_tostring(node, indent)
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
      if k ~= "node_type" and k~= "unparse" then
        str = str .. spaces1 .. k .. " = " ..
          ast_node_tostring(v, indent + 1) .. "," .. newline
      end
    end
    return str .. spaces .. ")"
  elseif terralib.islist(node) then
    local str = "{" .. newline
    for i, v in ipairs(node) do
      str = str .. spaces1 ..
        ast_node_tostring(v, indent + 1) .. "," .. newline
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
  print(tostring(self))
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
    if values[f] == nil then
      error(tostring(ctor) .. " does not require argument '" .. f .. "'", 2)
    end
    values[f] = v
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
    source = "",
    line = 0,
    offset = 0,
  }
end

-- Node Types (Unspecialized)

ast:inner("unspecialized", { "position" })

ast.unspecialized:leaf("Rules", { "rules" })
function ast.unspecialized.Rules:unparse()
  local str = ""
  for i = 1, #self.rules do
    str = str .. self.rules[i]:unparse()
  end
  return str
end
ast.unspecialized:leaf("Rule", { "selectors", "properties" })
function ast.unspecialized.Rule:unparse()
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

ast.unspecialized:leaf("Selector", { "elements" })
function ast.unspecialized.Selector:unparse()
  local str = self.elements[1]:unparse()
  for i = 2, #self.elements do
    str = str .. " " .. self.elements[i]:unparse()
  end
  return str
end

ast.unspecialized:leaf("Element", { "type", "name", "classes", "constraints" })
function ast.unspecialized.Element:unparse()
  local str = self.type
  for i = 1, #self.name do
    str = str .. "#" .. self.name[i]
  end
  for i = 1, #self.classes do
    str = str .. "." .. self.classes[i]
  end
  if #self.constraints > 0 then
    str = str .. "["
    str = str .. self.constraints[1]:unparse()
    for i = 2, #self.constraints do
      str = str .. " and " .. self.constraints[i]:unparse()
    end
    str = str .. "]"
  end
  return str
end

ast.unspecialized:leaf("Property", { "field", "value" })
function ast.unspecialized.Property:unparse()
  return self.field .. " : " .. self.value:unparse() .. ";"
end

ast.unspecialized:leaf("Constraint", { "field", "value" })
function ast.unspecialized.Constraint:unparse()
  return self.field .. " = " .. self.value:unparse()
end

ast.unspecialized:inner("expr")
ast.unspecialized.expr:leaf("Index", { "value", "index" })
function ast.unspecialized.expr.Index:unparse()
  if terralib.islist(self.index) then
    local index_str = self.index[1]:unparse()
    for i = 2, #self.index do
      index_str = index_str .. " and " .. self.index[i]:unparse()
    end
    return self.value:unparse() .. "[" .. index_str .. "]"
  else
    return self.value:unparse() .. "[" .. self.index:unparse() .. "]"
  end
end
ast.unspecialized.expr:leaf("Field", { "value", "field" })
function ast.unspecialized.expr.Field:unparse()
  return self.value:unparse() .. "." .. self.field
end

ast.unspecialized.expr:leaf("Constant", { "constant", "type" })
function ast.unspecialized.expr.Constant:unparse()
  if self.type == "keyword" then return self.constant
  else assert(type(self.constant) == "number") return tostring(self.constant) end
end
ast.unspecialized.expr:leaf("Variable", { "id" })
function ast.unspecialized.expr.Variable:unparse()
  return self.id
end

return ast
