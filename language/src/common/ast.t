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

-- ASTs

local data = require("common/data")

local ast = {}

local ast_factory = {}

function ast.make_factory(name)
  return setmetatable(
    {
      parent = false,
      children = terralib.newlist(),
      name = name,
      expected_fields = false,
      expected_field_set = false,
      print_collapsed = false,
      print_hidden = false,
    },
    ast_factory)
end

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

function ast_node:hash()
  local hash_value = "__ast_node_#" .. tostring(self.node_id)
  return hash_value
end

function ast.is_node(node)
  return type(node) == "table" and getmetatable(node) == ast_node
end

local function ast_node_tostring(node, indent, hide)
  local newline = "\n"
  local spaces = string.rep("  ", indent)
  local spaces1 = string.rep("  ", indent + 1)
  if ast.is_node(node) then
    local hidden = node.node_type.print_hidden
    if hide and hidden then return end

    local print_custom = node.node_type.print_custom
    if print_custom then
      if type(print_custom) == "string" then
        return print_custom
      else
        return print_custom(node)
      end
    end

    local collapsed = node.node_type.print_collapsed
    if collapsed then
      newline = ""
      spaces = ""
      spaces1 = ""
    end
    local str = tostring(node.node_type) .. "(" .. newline
    for k, v in pairs(node) do
      if k ~= "node_type" then
        local vstr = ast_node_tostring(v, indent + 1, hide)
        if vstr then
          str = str .. spaces1 .. k .. " = " .. vstr .. "," .. newline
        end
      end
    end
    return str .. spaces .. ")"
  elseif terralib.islist(node) then
    local str = "{" .. newline
    for i, v in ipairs(node) do
      local vstr = ast_node_tostring(v, indent + 1, hide)
      if vstr then
        str = str .. spaces1 .. vstr .. "," .. newline
      end
    end
    return str .. spaces .. "}"
  elseif type(node) == "string" then
    return string.format("%q", node)
  else
    return tostring(node)
  end
end

function ast_node:tostring(hide)
  if self.node_type.print_custom then
    if type(self.node_type.print_custom) == "string" then
      return self.node_type.print_custom
    else
      return self.node_type.print_custom(self)
    end
  else
    return ast_node_tostring(self, 0, hide)
  end
end

function ast_node:__tostring()
  return self:tostring(false)
end

function ast_node:printpretty(hide)
  print(self:tostring(hide))
end

function ast_node:is(node_type)
  return self.node_type:is(node_type)
end

function ast_node:type()
  return self.node_type
end

function ast_node:get_fields()
  local result = {}
  for k, v in pairs(self) do
    if k ~= "node_type" and k ~= "node_id" then
      result[k] = v
    end
  end
  return result
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

function ast_ctor:set_memoize()
  assert(not self.memoize_cache)
  if #self.expected_fields > 0 then
    self.memoize_cache = {}
  else
    self.memoize_cache = self({})
  end
  return self
end

function ast_ctor:set_print_custom(thunk)
  assert(not self.print_custom)
  self.print_custom = thunk
  return self
end

do
local next_ast_node_id = 0
function ast_ctor:__call(node)
  assert(type(node) == "table", tostring(self) .. " expected table")

  -- Normally, we assume we can co-opt the incoming table as the
  -- node. This is not true if the incoming node is itself an
  -- AST. (ASTs are not supposed to be mutable!) If so, copy the
  -- fields.
  local result = node
  if ast.is_node(node) then
    local copy = {}
    for k, v in pairs(node) do
      copy[k] = v
    end
    copy["node_type"] = nil
    copy["node_id"] = nil
    result = copy
  end

  -- Check that the supplied fields are necessary and sufficient.
  for _, f in ipairs(self.expected_fields) do
    if rawget(result, f) == nil then
      error(tostring(self) .. " missing required argument '" .. f .. "'", 2)
    end
  end
  for f, _ in pairs(result) do
    if rawget(self.expected_field_set, f) == nil then
      error(tostring(self) .. " does not require argument '" .. f .. "'", 2)
    end
  end

  -- Prepare the result to be returned.
  rawset(result, "node_type", self)

  local id = next_ast_node_id
  next_ast_node_id = next_ast_node_id + 1
  rawset(result, "node_id", id)

  setmetatable(result, ast_node)

  if self.memoize_cache then
    local cache = self.memoize_cache
    for i, f in ipairs(self.expected_fields) do
      local value = rawget(result, f)
      if not cache[value] then
        if i < #self.expected_fields then
          cache[value] = {}
        else
          cache[value] = result
        end
      end
      cache = cache[value]
    end
    if cache then
      assert(cache:is(self))
      return cache
    end
  end

  return result
end
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

function ast_factory:inner(ctor_name, expected_fields, print_collapsed, print_hidden)
  local fields = merge_fields(self.expected_fields, expected_fields)
  local ctor = setmetatable(
    {
      parent = self,
      children = terralib.newlist(),
      name = ctor_name,
      expected_fields = fields,
      expected_field_set = data.set(fields),
      print_collapsed = (print_collapsed == nil and self.print_collapsed) or print_collapsed or false,
      print_hidden = (print_hidden == nil and self.print_hidden) or print_hidden or false,
    }, ast_factory)

  assert(rawget(self, ctor_name) == nil,
         "multiple definitions of constructor " .. ctor_name)
  self[ctor_name] = ctor
  self.children:insert(ctor)
  return ctor
end

function ast_factory:leaf(ctor_name, expected_fields, print_collapsed, print_hidden)
  local fields = merge_fields(self.expected_fields, expected_fields)
  local ctor = setmetatable(
    {
      parent = self,
      children = terralib.newlist(),
      name = ctor_name,
      expected_fields = fields,
      expected_field_set = data.set(fields),
      print_collapsed = (print_collapsed == nil and self.print_collapsed) or print_collapsed or false,
      print_hidden = (print_hidden == nil and self.print_hidden) or print_hidden or false,
      print_custom = false,
      memoize_cache = false,
    }, ast_ctor)

  assert(rawget(self, ctor_name) == nil,
         "multiple definitions of constructor " .. ctor_name)
  self[ctor_name] = ctor
  self.children:insert(ctor)
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

-- Dispatch

local function check_dispatch_completeness(table, node_type)
  if table[node_type] then
    return true
  end

  if #node_type.children > 0 and data.all(unpack(
      node_type.children:map(
        function(child_type) return check_dispatch_completeness(table, child_type) end)))
  then
    return true
  end

  assert(false, "dispatch table is missing node type " .. tostring(node_type))
end

function ast.make_single_dispatch(table, required_cases)
  assert(table and required_cases)

  for _, parent_type in ipairs(required_cases) do
    check_dispatch_completeness(table, parent_type)
  end

  return function(cx)
    if cx then
      return function(node, ...)
        local node_type = node.node_type
        while node_type and not table[node_type] do
          node_type = node_type.parent
        end
        if table[node_type] then
          return table[node_type](cx, node, ...)
        else
          assert(false, "unexpected node type " .. tostring(node.node_type))
        end
      end
    else
      return function(node, ...)
        local node_type = node.node_type
        while node_type and not table[node_type] do
          node_type = node_type.parent
        end
        if table[node_type] then
          return table[node_type](node, ...)
        else
          assert(false, "unexpected node type " .. tostring(node.node_type))
        end
      end
    end
  end
end

-- Traversal

function ast.traverse_node_continuation(fn, node)
  local function continuation(node, continuing)
    if ast.is_node(node) then
      -- First entry: invoke the callback.
      if continuing == nil then
        fn(node, continuation)

      -- Second entry: (if true) continue to children.
      elseif continuing then
        for _, child in pairs(node) do
          continuation(child)
        end
      end
    elseif terralib.islist(node) then
      for _, child in ipairs(node) do
        continuation(child)
      end
    end
  end
  continuation(node)
end

function ast.map_node_continuation(fn, node)
  local function continuation(node, continuing)
    if ast.is_node(node) then
      -- First entry: invoke the callback.
      if continuing == nil then
        return fn(node, continuation)

      -- Second entry: (if true) continue to children.
      elseif continuing then
        local tmp = {}
        for k, child in pairs(node) do
          if k ~= "node_type" and k ~= "node_id" then
            tmp[k] = continuation(child)
          end
        end
        return node(tmp)
      end
    elseif terralib.islist(node) then
      local tmp = terralib.newlist()
      for _, child in ipairs(node) do
        tmp:insert(continuation(child))
      end
      return tmp
    end
    return node
  end
  return continuation(node)
end

function ast.traverse_node_postorder(fn, node)
  if ast.is_node(node) then
    for _, child in pairs(node) do
      ast.traverse_node_postorder(fn, child)
    end
    fn(node)
  elseif terralib.islist(node) then
    for _, child in ipairs(node) do
      ast.traverse_node_postorder(fn, child)
    end
  end
end

function ast.traverse_node_prepostorder(pre_fn, post_fn, node)
  if ast.is_node(node) then
    pre_fn(node)
    for k, child in pairs(node) do
      if k ~= "node_type" and k ~= "node_id" then
        ast.traverse_node_prepostorder(pre_fn, post_fn, child)
      end
    end
    post_fn(node)
  elseif terralib.islist(node) then
    for _, child in ipairs(node) do
      ast.traverse_node_prepostorder(pre_fn, post_fn, child)
    end
  end
end

function ast.map_node_postorder(fn, node)
  if ast.is_node(node) then
    local tmp = {}
    for k, child in pairs(node) do
      if k ~= "node_type" and k ~= "node_id" then
        tmp[k] = ast.map_node_postorder(fn, child)
      end
    end
    return fn(node(tmp))
  elseif terralib.islist(node) then
    local tmp = terralib.newlist()
    for _, child in ipairs(node) do
      tmp:insert(ast.map_node_postorder(fn, child))
    end
    return tmp
  end
  return node
end

function ast.map_node_prepostorder(pre_fn, post_fn, node)
  if ast.is_node(node) then
    local new_node = pre_fn(node)
    local tmp = {}
    for k, child in pairs(new_node) do
      if k ~= "node_type" and k ~= "node_id" then
        tmp[k] = ast.map_node_prepostorder(pre_fn, post_fn, child)
      end
    end
    return post_fn(new_node(tmp))
  elseif terralib.islist(node) then
    local tmp = terralib.newlist()
    for _, child in ipairs(node) do
      tmp:insert(ast.map_node_prepostorder(pre_fn, post_fn, child))
    end
    return tmp
  end
  return node
end

function ast.mapreduce_node_postorder(map_fn, reduce_fn, node, init)
  if ast.is_node(node) then
    local result = init
    for _, child in pairs(node) do
      result = reduce_fn(
        result,
        ast.mapreduce_node_postorder(map_fn, reduce_fn, child, init))
    end
    return reduce_fn(result, map_fn(node))
  elseif terralib.islist(node) then
    local result = init
    for _, child in ipairs(node) do
      result = reduce_fn(
        result,
        ast.mapreduce_node_postorder(map_fn, reduce_fn, child, init))
    end
    return result
  end
  return init
end

function ast.traverse_expr_postorder(fn, node)
  ast.traverse_node_postorder(
    function(child)
      if rawget(child, "expr_type") then
        fn(child)
      end
    end,
    node)
end

-- Location

ast.location = ast.make_factory("ast.location")
ast.location:leaf("Position", {"line", "offset"}, true)
ast.location:leaf("Span", {"source", "start", "stop"}, false, true)

-- Helpers for extracting location from token stream.
local function position_from_start(token)
  return ast.location.Position {
    line = token.linenumber,
    offset = token.offset
  }
end

local function position_from_stop(token)
  return position_from_start(token)
end

function ast.save(p)
  return position_from_start(p:cur())
end

function ast.span(start, p)
  return ast.location.Span {
    source = p.source,
    start = start,
    stop = position_from_stop(p:cur()),
  }
end

function ast.empty_span(p)
  return ast.location.Span {
    source = p.source,
    start = ast.location.Position { line = 0, offset = 0 },
    stop = ast.location.Position { line = 0, offset = 0 },
  }
end

function ast.trivial_span()
  return ast.location.Span {
    source = "",
    start = ast.location.Position { line = 0, offset = 0 },
    stop = ast.location.Position { line = 0, offset = 0 },
  }
end

return ast
