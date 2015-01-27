-- Copyright 2015 Stanford University
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

local function ast_node_tostring(node, indent)
  if type(node) == "table" and getmetatable(node) == ast_node then
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
  return self.node_type == node_type
end

function ast_node:type()
  return self.node_type
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
  return "ast." .. ctor.factory.name .. "." .. ctor.name
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


local function AST_factory(name)
  local factory = {name = name}
  setmetatable(factory, ast_factory)

  return factory
end

-- Node Types (Unspecialized)

ast.unspecialized = AST_factory("unspecialized")

ast.unspecialized("ExprID", {"name"})
ast.unspecialized("ExprEscape", {"expr"})
ast.unspecialized("ExprFieldAccess", {"value", "field_name"})
ast.unspecialized("ExprIndexAccess", {"value", "index"})
ast.unspecialized("ExprMethodCall", {"value", "method_name", "args"})
ast.unspecialized("ExprCall", {"fn", "args"})
ast.unspecialized("ExprCtor", {"fields"})
ast.unspecialized("ExprCtorListField", {"value"})
ast.unspecialized("ExprCtorRecField", {"name_expr", "value"})
ast.unspecialized("ExprConstant", {"value", "expr_type"})
ast.unspecialized("ExprRawContext", {})
ast.unspecialized("ExprRawFields", {"region"})
ast.unspecialized("ExprRawPhysical", {"region"})
ast.unspecialized("ExprRawRuntime", {})
ast.unspecialized("ExprIsnull", {"pointer"})
ast.unspecialized("ExprNew", {"pointer_type_expr"})
ast.unspecialized("ExprNull", {"pointer_type_expr"})
ast.unspecialized("ExprRegion", {"element_type_expr", "size"})
ast.unspecialized("ExprPartition", {"disjointness_expr", "region_type_expr",
                                    "coloring"})
ast.unspecialized("ExprUnary", {"op", "rhs"})
ast.unspecialized("ExprBinary", {"op", "lhs", "rhs"})
ast.unspecialized("ExprDeref", {"value"})

ast.unspecialized("Block", {"stats"})

ast.unspecialized("StatIf", {"cond", "then_block", "elseif_blocks",
                             "else_block"})
ast.unspecialized("StatElseif", {"cond", "block"})
ast.unspecialized("StatWhile", {"cond", "block"})
ast.unspecialized("StatForNum", {"name", "type_expr", "values", "block"})
ast.unspecialized("StatForList", {"name", "type_expr", "value", "block"})
ast.unspecialized("StatRepeat", {"block", "until_cond"})
ast.unspecialized("StatBlock", {"block"})
ast.unspecialized("StatVar", {"var_names", "type_exprs", "values"})
ast.unspecialized("StatVarUnpack", {"var_names", "fields", "value"})
ast.unspecialized("StatReturn", {"value"})
ast.unspecialized("StatBreak", {})
ast.unspecialized("StatAssignment", {"lhs", "rhs"})
ast.unspecialized("StatReduce", {"op", "lhs", "rhs"})
ast.unspecialized("StatExpr", {"expr"})

ast.unspecialized("StatTask", {"name", "params", "return_type_expr",
                               "privilege_exprs", "body"})
ast.unspecialized("StatTaskParam", {"param_name", "type_expr"})
ast.unspecialized("StatTaskPrivilege", {"privilege", "op", "regions"})
ast.unspecialized("StatTaskPrivilegeRegion", {"region_name", "fields"})
ast.unspecialized("StatTaskPrivilegeRegionField", {"field_name", "fields"})
ast.unspecialized("StatFspace", {"name", "params", "fields", "constraints"})
ast.unspecialized("StatFspaceParam", {"param_name", "type_expr"})
ast.unspecialized("StatFspaceField", {"field_name", "type_expr"})
ast.unspecialized("StatFspaceConstraint", {"lhs", "op", "rhs"})

-- Node Types (Specialized)

ast.specialized = AST_factory("specialized")

ast.specialized("ExprID", {"value"})
ast.specialized("ExprFieldAccess", {"value", "field_name"})
ast.specialized("ExprIndexAccess", {"value", "index"})
ast.specialized("ExprMethodCall", {"value", "method_name", "args"})
ast.specialized("ExprCall", {"fn", "args"})
ast.specialized("ExprCast", {"fn", "args"})
ast.specialized("ExprCtor", {"fields", "named"})
ast.specialized("ExprCtorListField", {"value"})
ast.specialized("ExprCtorRecField", {"name", "value"})
ast.specialized("ExprConstant", {"value", "expr_type"})
ast.specialized("ExprRawContext", {})
ast.specialized("ExprRawFields", {"region"})
ast.specialized("ExprRawPhysical", {"region"})
ast.specialized("ExprRawRuntime", {})
ast.specialized("ExprIsnull", {"pointer"})
ast.specialized("ExprNew", {"pointer_type", "region"})
ast.specialized("ExprNull", {"pointer_type"})
ast.specialized("ExprRegion", {"element_type", "size", "expr_type"})
ast.specialized("ExprPartition", {"disjointness", "region",
                                  "coloring", "expr_type"})
ast.specialized("ExprFunction", {"value"})
ast.specialized("ExprUnary", {"op", "rhs"})
ast.specialized("ExprBinary", {"op", "lhs", "rhs"})
ast.specialized("ExprDeref", {"value"})
ast.specialized("ExprLuaTable", {"value"})

ast.specialized("Block", {"stats"})

ast.specialized("StatIf", {"cond", "then_block", "elseif_blocks", "else_block"})
ast.specialized("StatElseif", {"cond", "block"})
ast.specialized("StatWhile", {"cond", "block"})
ast.specialized("StatForNum", {"symbol", "values", "block"})
ast.specialized("StatForList", {"symbol", "value", "block"})
ast.specialized("StatRepeat", {"block", "until_cond"})
ast.specialized("StatBlock", {"block"})
ast.specialized("StatVar", {"symbols", "values"})
ast.specialized("StatVarUnpack", {"symbols", "fields", "value"})
ast.specialized("StatReturn", {"value"})
ast.specialized("StatBreak", {})
ast.specialized("StatAssignment", {"lhs", "rhs"})
ast.specialized("StatReduce", {"op", "lhs", "rhs"})
ast.specialized("StatExpr", {"expr"})

ast.specialized("StatTask", {"name", "params", "return_type", "privileges",
                             "body", "prototype"})
ast.specialized("StatTaskParam", {"symbol"})
ast.specialized("StatFspace", {"name", "fspace"})

-- Node Types (Typed)

ast.typed = AST_factory("typed")

ast.typed("ExprID", {"value", "expr_type"})
ast.typed("ExprFieldAccess", {"value", "field_name", "expr_type"})
ast.typed("ExprIndexAccess", {"value", "index", "expr_type"})
ast.typed("ExprMethodCall", {"value", "method_name", "args", "expr_type"})
ast.typed("ExprCall", {"fn", "args", "expr_type"})
ast.typed("ExprCast", {"fn", "arg", "expr_type"})
ast.typed("ExprCtor", {"fields", "named", "expr_type"})
ast.typed("ExprCtorListField", {"value"})
ast.typed("ExprCtorRecField", {"name", "value"})
ast.typed("ExprRawContext", {"expr_type"})
ast.typed("ExprRawFields", {"region", "fields", "expr_type"})
ast.typed("ExprRawPhysical", {"region", "fields", "expr_type"})
ast.typed("ExprRawRuntime", {"expr_type"})
ast.typed("ExprIsnull", {"pointer", "expr_type"})
ast.typed("ExprNew", {"pointer_type", "region", "expr_type"})
ast.typed("ExprNull", {"pointer_type", "expr_type"})
ast.typed("ExprRegion", {"element_type", "size", "expr_type"})
ast.typed("ExprPartition", {"disjointness", "region",
                            "coloring", "expr_type"})
ast.typed("ExprConstant", {"value", "expr_type"})
ast.typed("ExprFunction", {"value", "expr_type"})
ast.typed("ExprUnary", {"op", "rhs", "expr_type"})
ast.typed("ExprBinary", {"op", "lhs", "rhs", "expr_type"})
ast.typed("ExprDeref", {"value", "expr_type"})

ast.typed("Block", {"stats"})

ast.typed("StatIf", {"cond", "then_block", "elseif_blocks", "else_block"})
ast.typed("StatElseif", {"cond", "block"})
ast.typed("StatWhile", {"cond", "block"})
ast.typed("StatForNum", {"symbol", "values", "block"})
ast.typed("StatForList", {"symbol", "value", "block"})
ast.typed("StatRepeat", {"block", "until_cond"})
ast.typed("StatBlock", {"block"})
ast.typed("StatVar", {"symbols", "types", "values"})
ast.typed("StatVarUnpack", {"symbols", "fields", "field_types", "value"})
ast.typed("StatReturn", {"value"})
ast.typed("StatBreak", {})
ast.typed("StatAssignment", {"lhs", "rhs"})
ast.typed("StatReduce", {"op", "lhs", "rhs"})
ast.typed("StatExpr", {"expr"})

ast.typed("StatTask", {"name", "params", "return_type", "privileges", "body",
                       "prototype"})
ast.typed("StatTaskParam", {"symbol", "param_type"})
ast.typed("StatFspace", {"name", "fspace"})

return ast
