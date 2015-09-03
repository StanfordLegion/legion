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

-- Legion AST

local ast = {}

-- Nodes

local ast_node = {}

function ast_node:__index(field)
  local value = ast_node[field]
  if value ~= nil then
    return value
  end
  local node_type = tostring(rawget(self, "node_type")) or "(unknown)"
  error(node_type .. " has no field '" .. field .. "'", 2)
end

function ast.is_node(node)
  return type(node) == "table" and getmetatable(node) == ast_node
end

local function ast_node_tostring(node, indent)
  if ast.is_node(node) then
    local str = tostring(node.node_type) .. "(\n"
    for k, v in pairs(node) do
      if k ~= "node_type" then
        str = str .. string.rep("  ", indent + 1) .. k .. " = " ..
          ast_node_tostring(v, indent + 1) .. ",\n"
      end
    end
    return str .. string.rep("  ", indent) .. ")"
  elseif terralib.islist(node) then
    local str = "{\n"
    for i, v in ipairs(node) do
      str = str .. string.rep("  ", indent + 1) ..
        ast_node_tostring(v, indent + 1) .. ",\n"
    end
    return str .. string.rep("  ", indent) .. "}"
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

function ast_node:is(node_type)
  return self.node_type == node_type or self.node_type.factory == node_type
end

function ast_node:type()
  return self.node_type
end

ast_node.__call = function(node, fields_to_update)
  local ctor = rawget(node, "node_type")
  local values = {}
  for _, f in ipairs(ctor.expected_fields) do
    values[f] = node[f]
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

ast_ctor.__call = function(ctor, node)
  assert(type(node) == "table", tostring(ctor) .. " expected table")
  for i, f in ipairs(ctor.expected_fields) do
    if rawget(node, f) == nil then
      error(tostring(ctor) .. " missing required argument '" .. f .. "'", 2)
    end
  end
  rawset(node, "node_type", ctor)
  setmetatable(node, ast_node)
  return node
end

ast_ctor.__tostring = function(ctor)
  return ctor.factory.name .. "." .. ctor.name
end

-- Factories

local ast_factory = {}

ast_factory.__call = function(factory, ctor_name, expected_fields)
  local ctor = {
    factory = factory,
    name = ctor_name,
    expected_fields = expected_fields,
  }
  setmetatable(ctor, ast_ctor)

  assert(rawget(factory, ctor_name) == nil,
         "multiple definitions of constructor " .. ctor_name)
  factory[ctor_name] = ctor
  return ctor
end


function ast.factory(name)
  local factory = {name = name}
  setmetatable(factory, ast_factory)

  return factory
end

-- Traversal

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

function ast.map_node_postorder(fn, node)
  if ast.is_node(node) then
    local tmp = {}
    for k, child in pairs(node) do
      if k ~= "node_type" then
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

ast.location = ast.factory("location")

ast.location("Position", {"line", "offset"})
ast.location("Span", {"source", "start", "stop"})

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

-- Node Types (Unspecialized)

ast.unspecialized = ast.factory("ast.unspecialized")

ast.unspecialized("ExprID", {"name", "span"})
ast.unspecialized("ExprEscape", {"expr", "span"})
ast.unspecialized("ExprFieldAccess", {"value", "field_names", "span"})
ast.unspecialized("ExprIndexAccess", {"value", "index", "span"})
ast.unspecialized("ExprMethodCall", {"value", "method_name", "args", "span"})
ast.unspecialized("ExprCall", {"fn", "args", "inline", "span"})
ast.unspecialized("ExprCtor", {"fields", "span"})
ast.unspecialized("ExprCtorListField", {"value", "span"})
ast.unspecialized("ExprCtorRecField", {"name_expr", "value", "span"})
ast.unspecialized("ExprConstant", {"value", "expr_type", "span"})
ast.unspecialized("ExprRawContext", {"span"})
ast.unspecialized("ExprRawFields", {"region", "span"})
ast.unspecialized("ExprRawPhysical", {"region", "span"})
ast.unspecialized("ExprRawRuntime", {"span"})
ast.unspecialized("ExprRawValue", {"value", "span"})
ast.unspecialized("ExprIsnull", {"pointer", "span"})
ast.unspecialized("ExprNew", {"pointer_type_expr", "span"})
ast.unspecialized("ExprNull", {"pointer_type_expr", "span"})
ast.unspecialized("ExprDynamicCast", {"type_expr", "value", "span"})
ast.unspecialized("ExprStaticCast", {"type_expr", "value", "span"})
ast.unspecialized("ExprIspace", {"index_type_expr", "extent",
                                 "start", "span"})
ast.unspecialized("ExprRegion", {"ispace", "fspace_type_expr", "span"})
ast.unspecialized("ExprPartition", {"disjointness_expr", "region_type_expr",
                                    "coloring", "span"})
ast.unspecialized("ExprCrossProduct", {"arg_type_exprs", "span"})
ast.unspecialized("ExprUnary", {"op", "rhs", "span"})
ast.unspecialized("ExprBinary", {"op", "lhs", "rhs", "span"})
ast.unspecialized("ExprDeref", {"value", "span"})

ast.unspecialized("Block", {"stats", "span"})

ast.unspecialized("StatIf", {"cond", "then_block", "elseif_blocks",
                             "else_block", "span"})
ast.unspecialized("StatElseif", {"cond", "block", "span"})
ast.unspecialized("StatWhile", {"cond", "block", "span"})
ast.unspecialized("StatForNum", {"name", "type_expr", "values", "block",
                                 "parallel", "span"})
ast.unspecialized("StatForList", {"name", "type_expr", "value", "block",
                                  "vectorize", "span"})
ast.unspecialized("StatRepeat", {"block", "until_cond", "span"})
ast.unspecialized("StatBlock", {"block", "span"})
ast.unspecialized("StatVar", {"var_names", "type_exprs", "values", "span"})
ast.unspecialized("StatVarUnpack", {"var_names", "fields", "value", "span"})
ast.unspecialized("StatReturn", {"value", "span"})
ast.unspecialized("StatBreak", {"span"})
ast.unspecialized("StatAssignment", {"lhs", "rhs", "span"})
ast.unspecialized("StatReduce", {"op", "lhs", "rhs", "span"})
ast.unspecialized("StatExpr", {"expr", "span"})

ast.unspecialized("Constraint", {"lhs", "op", "rhs", "span"})
ast.unspecialized("Privilege", {"privilege", "op", "regions", "span"})
ast.unspecialized("PrivilegeRegion", {"region_name", "fields", "span"})
ast.unspecialized("PrivilegeRegionField", {"field_name", "fields", "span"})
ast.unspecialized("StatTask", {"name", "params", "return_type_expr",
                               "privileges", "constraints", "body",
                               "inline", "cuda", "span"})
ast.unspecialized("StatTaskParam", {"param_name", "type_expr", "span"})
ast.unspecialized("StatFspace", {"name", "params", "fields", "constraints",
                                 "span"})
ast.unspecialized("StatFspaceParam", {"param_name", "type_expr", "span"})
ast.unspecialized("StatFspaceField", {"field_name", "type_expr", "span"})

-- Node Types (Specialized)

ast.specialized = ast.factory("ast.specialized")

ast.specialized("ExprID", {"value", "span"})
ast.specialized("ExprFieldAccess", {"value", "field_name", "span"})
ast.specialized("ExprIndexAccess", {"value", "index", "span"})
ast.specialized("ExprMethodCall", {"value", "method_name", "args", "span"})
ast.specialized("ExprCall", {"fn", "args", "inline", "span"})
ast.specialized("ExprCast", {"fn", "args", "span"})
ast.specialized("ExprCtor", {"fields", "named", "span"})
ast.specialized("ExprCtorListField", {"value", "span"})
ast.specialized("ExprCtorRecField", {"name", "value", "span"})
ast.specialized("ExprConstant", {"value", "expr_type", "span"})
ast.specialized("ExprRawContext", {"span"})
ast.specialized("ExprRawFields", {"region", "span"})
ast.specialized("ExprRawPhysical", {"region", "span"})
ast.specialized("ExprRawRuntime", {"span"})
ast.specialized("ExprRawValue", {"value", "span"})
ast.specialized("ExprIsnull", {"pointer", "span"})
ast.specialized("ExprNew", {"pointer_type", "region", "span"})
ast.specialized("ExprNull", {"pointer_type", "span"})
ast.specialized("ExprDynamicCast", {"value", "expr_type", "span"})
ast.specialized("ExprStaticCast", {"value", "expr_type", "span"})
ast.specialized("ExprIspace", {"index_type", "extent", "start",
                               "expr_type", "span"})
ast.specialized("ExprRegion", {"ispace", "ispace_symbol", "fspace_type", "expr_type", "span"})
ast.specialized("ExprPartition", {"disjointness", "region",
                                  "coloring", "expr_type", "span"})
ast.specialized("ExprCrossProduct", {"args", "expr_type", "span"})
ast.specialized("ExprFunction", {"value", "span"})
ast.specialized("ExprUnary", {"op", "rhs", "span"})
ast.specialized("ExprBinary", {"op", "lhs", "rhs", "span"})
ast.specialized("ExprDeref", {"value", "span"})
ast.specialized("ExprLuaTable", {"value", "span"})

ast.specialized("Block", {"stats", "span"})

ast.specialized("StatIf", {"cond", "then_block", "elseif_blocks", "else_block",
                           "span"})
ast.specialized("StatElseif", {"cond", "block", "span"})
ast.specialized("StatWhile", {"cond", "block", "span"})
ast.specialized("StatForNum", {"symbol", "values", "block", "parallel",
                               "span"})
ast.specialized("StatForList", {"symbol", "value", "block", "vectorize",
                                "span"})
ast.specialized("StatRepeat", {"block", "until_cond", "span"})
ast.specialized("StatBlock", {"block", "span"})
ast.specialized("StatVar", {"symbols", "values", "span"})
ast.specialized("StatVarUnpack", {"symbols", "fields", "value", "span"})
ast.specialized("StatReturn", {"value", "span"})
ast.specialized("StatBreak", {"span"})
ast.specialized("StatAssignment", {"lhs", "rhs", "span"})
ast.specialized("StatReduce", {"op", "lhs", "rhs", "span"})
ast.specialized("StatExpr", {"expr", "span"})

ast.specialized("StatTask", {"name", "params", "return_type", "privileges",
                             "constraints", "body", "prototype", "inline",
                             "cuda", "span"})
ast.specialized("StatTaskParam", {"symbol", "span"})
ast.specialized("StatFspace", {"name", "fspace", "span"})

-- Node Types (Typed)

ast.typed = ast.factory("ast.typed")

ast.typed("ExprInternal", {"value", "expr_type"}) -- internal use only

ast.typed("ExprID", {"value", "expr_type", "span"})
ast.typed("ExprFieldAccess", {"value", "field_name", "expr_type", "span"})
ast.typed("ExprIndexAccess", {"value", "index", "expr_type", "span"})
ast.typed("ExprMethodCall", {"value", "method_name", "args", "expr_type", "span"})
ast.typed("ExprCall", {"fn", "args", "expr_type", "inline", "span"})
ast.typed("ExprCast", {"fn", "arg", "expr_type", "span"})
ast.typed("ExprCtor", {"fields", "named", "expr_type", "span"})
ast.typed("ExprCtorListField", {"value", "expr_type", "span"})
ast.typed("ExprCtorRecField", {"name", "value", "expr_type", "span"})
ast.typed("ExprRawContext", {"expr_type", "span"})
ast.typed("ExprRawFields", {"region", "fields", "expr_type", "span"})
ast.typed("ExprRawPhysical", {"region", "fields", "expr_type", "span"})
ast.typed("ExprRawRuntime", {"expr_type", "span"})
ast.typed("ExprRawValue", {"value", "expr_type", "span"})
ast.typed("ExprIsnull", {"pointer", "expr_type", "span"})
ast.typed("ExprNew", {"pointer_type", "region", "expr_type", "span"})
ast.typed("ExprNull", {"pointer_type", "expr_type", "span"})
ast.typed("ExprDynamicCast", {"value", "expr_type", "span"})
ast.typed("ExprStaticCast", {"value", "parent_region_map", "expr_type", "span"})
ast.typed("ExprIspace", {"index_type", "extent", "start",
                         "expr_type", "span"})
ast.typed("ExprRegion", {"ispace", "fspace_type", "expr_type", "span"})
ast.typed("ExprPartition", {"disjointness", "region",
                            "coloring", "expr_type", "span"})
ast.typed("ExprCrossProduct", {"args", "expr_type", "span"})
ast.typed("ExprConstant", {"value", "expr_type", "span"})
ast.typed("ExprFunction", {"value", "expr_type", "span"})
ast.typed("ExprUnary", {"op", "rhs", "expr_type", "span"})
ast.typed("ExprBinary", {"op", "lhs", "rhs", "expr_type", "span"})
ast.typed("ExprDeref", {"value", "expr_type", "span"})
ast.typed("ExprFuture", {"value", "expr_type", "span"})
ast.typed("ExprFutureGetResult", {"value", "expr_type", "span"})

ast.typed("Block", {"stats", "span"})

ast.typed("StatIf", {"cond", "then_block", "elseif_blocks", "else_block", "span"})
ast.typed("StatElseif", {"cond", "block", "span"})
ast.typed("StatWhile", {"cond", "block", "span"})
ast.typed("StatForNum", {"symbol", "values", "block", "parallel", "span"})
ast.typed("StatForList", {"symbol", "value", "block", "vectorize", "span"})
ast.typed("StatForListVectorized", {"symbol", "value", "block", "orig_block",
                                    "vector_width", "span"})
ast.typed("StatRepeat", {"block", "until_cond", "span"})
ast.typed("StatBlock", {"block", "span"})
ast.typed("StatIndexLaunch", {"symbol", "domain", "call", "reduce_lhs",
                              "reduce_op", "args_provably", "span"})
ast.typed("StatIndexLaunchArgsProvably", {"invariant", "variant"})
ast.typed("StatVar", {"symbols", "types", "values", "span"})
ast.typed("StatVarUnpack", {"symbols", "fields", "field_types", "value", "span"})
ast.typed("StatReturn", {"value", "span"})
ast.typed("StatBreak", {"span"})
ast.typed("StatAssignment", {"lhs", "rhs", "span"})
ast.typed("StatReduce", {"op", "lhs", "rhs", "span"})
ast.typed("StatExpr", {"expr", "span"})
ast.typed("StatMapRegions", {"region_types"})
ast.typed("StatUnmapRegions", {"region_types"})

ast.typed("StatTask", {"name", "params", "return_type", "privileges",
                       "constraints", "body", "config_options",
                       "region_divergence", "prototype", "inline", "cuda",
                       "span"})
ast.typed("StatTaskParam", {"symbol", "param_type", "span"})
ast.typed("StatTaskConfigOptions", {"leaf", "inner", "idempotent"})
ast.typed("StatFspace", {"name", "fspace", "span"})

return ast
