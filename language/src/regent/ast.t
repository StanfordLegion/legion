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

-- Regent AST

local data = require("common/data")
local common_ast = require("common/ast")

local ast = common_ast.make_factory("ast")

-- Re-export entries from common AST library.
for k, v in pairs(common_ast) do
  ast[k] = v
end

-- Traversal

function ast.flatmap_node_continuation(fn, node)
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
            local is_src_list = terralib.islist(child)
            local is_dst_list = terralib.islist(tmp[k])
            assert((is_src_list and is_dst_list) or (not is_src_list and not is_dst_list),
                   "flatmap only flattens a list of statements")
          end
        end
        return node(tmp)
      end
    elseif terralib.islist(node) then
      local tmp = terralib.newlist()
      for _, child in ipairs(node) do
        child = continuation(child)
        if terralib.islist(child) then
          tmp:insertall(child)
        else
          tmp:insert(child)
        end
      end
      return tmp
    end
    return node
  end
  return continuation(node)
end

function ast.flatmap_node_postorder(fn, node)
  if ast.is_node(node) then
    local tmp = {}
    for k, child in pairs(node) do
       if k ~= "node_type" and k ~= "node_id" then
        tmp[k] = ast.flatmap_node_postorder(fn, child)
        local is_src_list = terralib.islist(child)
        local is_dst_list = terralib.islist(tmp[k])
        assert((is_src_list and is_dst_list) or (not is_src_list and not is_dst_list),
               "flatmap only flattens a list of statements")
      end
    end
    return fn(node(tmp))
  elseif terralib.islist(node) then
    local tmp = terralib.newlist()
    for _, child in ipairs(node) do
      local child = ast.flatmap_node_postorder(fn, child)
      if terralib.islist(child) then
        tmp:insertall(child)
      else
        tmp:insert(child)
      end
    end
    return tmp
  end
  return node
end

-- Annotation

ast:inner("annotation")

-- Annotation: Dispositions
ast.annotation:leaf("Allow", {"value"}, true)
ast.annotation:leaf("Demand", {"value"}, true)
ast.annotation:leaf("Forbid", {"value"}, true)

-- Annotation: Values
ast.annotation:leaf("Unroll", {"value"}, true)

-- Annotation: Sets
ast.annotation:leaf("Set", {"cuda", "external", "inline", "inner", "leaf",
                            "openmp", "optimize", "parallel", "spmd", "trace",
                            "vectorize"},
                    false, true)

function ast.default_annotations()
  local allow = ast.annotation.Allow { value = false }
  return ast.annotation.Set {
    cuda = allow,
    external = allow,
    inline = allow,
    inner = allow,
    leaf = allow,
    openmp = allow,
    optimize = allow,
    parallel = allow,
    spmd = allow,
    trace = allow,
    vectorize = allow,
  }
end

-- Calling Conventions

ast:inner("convention_kind")
ast.convention_kind:leaf("Padded"):set_memoize():set_print_custom("padded")
ast.convention_kind:leaf("Packed"):set_memoize():set_print_custom("packed")

ast:inner("convention")
ast.convention:leaf("Regent"):set_memoize():set_print_custom("regent")
ast.convention:leaf("Manual", {"params"}):set_memoize():set_print_custom(
  function(node) return "manual(" .. tostring(node.params) .. ")" end)

-- Kinds: Constraints, Privileges, Coherence, Flags, Conditions, Disjointness

ast:inner("constraint_kind")
ast.constraint_kind:leaf("Subregion"):set_memoize():set_print_custom("<=")
ast.constraint_kind:leaf("Disjointness"):set_memoize():set_print_custom("*")

ast:inner("privilege_kind")
ast.privilege_kind:leaf("Reads"):set_memoize():set_print_custom("reads")
ast.privilege_kind:leaf("Writes"):set_memoize():set_print_custom("writes")
ast.privilege_kind:leaf("Reduces", {"op"}):set_memoize():set_print_custom(
  function(node) return "reduces " .. tostring(node.op) end)

ast:inner("coherence_kind")
ast.coherence_kind:leaf("Exclusive"):set_memoize():set_print_custom("exclusive")
ast.coherence_kind:leaf("Atomic"):set_memoize():set_print_custom("atomic")
ast.coherence_kind:leaf("Simultaneous"):set_memoize():set_print_custom(
  "simultaneous")
ast.coherence_kind:leaf("Relaxed"):set_memoize():set_print_custom("relaxed")

ast:inner("flag_kind")
ast.flag_kind:leaf("NoAccessFlag"):set_memoize():set_print_custom(
  "no_access_flag")

ast:inner("condition_kind")
ast.condition_kind:leaf("Arrives"):set_memoize():set_print_custom("arrives")
ast.condition_kind:leaf("Awaits"):set_memoize():set_print_custom("awaits")

ast:inner("disjointness_kind")
ast.disjointness_kind:leaf("Aliased"):set_memoize():set_print_custom("aliased")
ast.disjointness_kind:leaf("Disjoint"):set_memoize():set_print_custom(
  "disjoint")

ast:inner("fence_kind")
ast.fence_kind:leaf("Execution"):set_memoize():set_print_custom("__execution")
ast.fence_kind:leaf("Mapping"):set_memoize():set_print_custom("__mapping")

-- Constraints

ast:inner("constraint")
ast.constraint:leaf("Constraint", {"lhs", "rhs", "op"})

-- Privileges

ast:inner("privilege")
ast.privilege:leaf("Privilege", {"privilege", "region", "field_path"})

-- Node Types (Unspecialized)

ast:inner("unspecialized", {"span"})

ast.unspecialized:leaf("FieldNames", {"names_expr"})

ast.unspecialized:inner("region")
ast.unspecialized.region:leaf("Bare", {"region_name"})
ast.unspecialized.region:leaf("Root", {"region_name", "fields"})
ast.unspecialized.region:leaf("Field", {"field_name", "fields"})

ast.unspecialized:leaf("Constraint", {"lhs", "op", "rhs"})

ast.unspecialized:leaf("Privilege", {"privileges", "regions"})

ast.unspecialized:leaf("Coherence", {"coherence_modes", "regions"})

ast.unspecialized:leaf("Flag", {"flags", "regions"})

ast.unspecialized:leaf("ConditionVariable", {"name"})
ast.unspecialized:leaf("Condition", {"conditions", "variables"})

ast.unspecialized:leaf("Effect", {"expr"})

ast.unspecialized:inner("expr", {"annotations"})
ast.unspecialized.expr:leaf("ID", {"name"})
ast.unspecialized.expr:leaf("Escape", {"expr"})
ast.unspecialized.expr:leaf("FieldAccess", {"value", "field_names"})
ast.unspecialized.expr:leaf("IndexAccess", {"value", "index"})
ast.unspecialized.expr:leaf("MethodCall", {"value", "method_name", "args"})
ast.unspecialized.expr:leaf("Call", {"fn", "args", "conditions"})
ast.unspecialized.expr:leaf("Ctor", {"fields"})
ast.unspecialized.expr:leaf("CtorListField", {"value"})
ast.unspecialized.expr:leaf("CtorRecField", {"name_expr", "value"})
ast.unspecialized.expr:leaf("Constant", {"value", "expr_type"})
ast.unspecialized.expr:leaf("RawContext")
ast.unspecialized.expr:leaf("RawFields", {"region"})
ast.unspecialized.expr:leaf("RawPhysical", {"region"})
ast.unspecialized.expr:leaf("RawRuntime")
ast.unspecialized.expr:leaf("RawValue", {"value"})
ast.unspecialized.expr:leaf("Isnull", {"pointer"})
ast.unspecialized.expr:leaf("New", {"pointer_type_expr", "extent"})
ast.unspecialized.expr:leaf("Null", {"pointer_type_expr"})
ast.unspecialized.expr:leaf("DynamicCast", {"type_expr", "value"})
ast.unspecialized.expr:leaf("StaticCast", {"type_expr", "value"})
ast.unspecialized.expr:leaf("UnsafeCast", {"type_expr", "value"})
ast.unspecialized.expr:leaf("Ispace", {"index_type_expr", "extent", "start"})
ast.unspecialized.expr:leaf("Region", {"ispace", "fspace_type_expr"})
ast.unspecialized.expr:leaf("Partition", {"disjointness", "region", "coloring",
                                          "colors"})
ast.unspecialized.expr:leaf("PartitionEqual", {"region", "colors"})
ast.unspecialized.expr:leaf("PartitionByField", {"region", "colors"})
ast.unspecialized.expr:leaf("Image", {"parent", "partition", "region"})
ast.unspecialized.expr:leaf("Preimage", {"parent", "partition", "region"})
ast.unspecialized.expr:leaf("CrossProduct", {"args"})
ast.unspecialized.expr:leaf("CrossProductArray", {"lhs", "disjointness", "colorings"})
ast.unspecialized.expr:leaf("ListSlicePartition", {"partition", "indices"})
ast.unspecialized.expr:leaf("ListDuplicatePartition", {"partition", "indices"})
ast.unspecialized.expr:leaf("ListCrossProduct", {"lhs", "rhs", "shallow"})
ast.unspecialized.expr:leaf("ListCrossProductComplete", {"lhs", "product"})
ast.unspecialized.expr:leaf("ListPhaseBarriers", {"product"})
ast.unspecialized.expr:leaf("ListInvert", {"rhs", "product", "barriers"})
ast.unspecialized.expr:leaf("ListRange", {"start", "stop"})
ast.unspecialized.expr:leaf("ListIspace", {"ispace"})
ast.unspecialized.expr:leaf("ListFromElement", {"list", "value"})
ast.unspecialized.expr:leaf("PhaseBarrier", {"value"})
ast.unspecialized.expr:leaf("DynamicCollective", {"value_type_expr", "op", "arrivals"})
ast.unspecialized.expr:leaf("DynamicCollectiveGetResult", {"value"})
ast.unspecialized.expr:leaf("Advance", {"value"})
ast.unspecialized.expr:leaf("Adjust", {"barrier", "value"})
ast.unspecialized.expr:leaf("Arrive", {"barrier", "value"})
ast.unspecialized.expr:leaf("Await", {"barrier"})
ast.unspecialized.expr:leaf("Copy", {"src", "dst", "op", "conditions"})
ast.unspecialized.expr:leaf("Fill", {"dst", "value", "conditions"})
ast.unspecialized.expr:leaf("Acquire", {"region", "conditions"})
ast.unspecialized.expr:leaf("Release", {"region", "conditions"})
ast.unspecialized.expr:leaf("AttachHDF5", {"region", "filename", "mode"})
ast.unspecialized.expr:leaf("DetachHDF5", {"region"})
ast.unspecialized.expr:leaf("AllocateScratchFields", {"region"})
ast.unspecialized.expr:leaf("WithScratchFields", {"region", "field_ids"})
ast.unspecialized.expr:leaf("RegionRoot", {"region", "fields"})
ast.unspecialized.expr:leaf("Condition", {"conditions", "values"})
ast.unspecialized.expr:leaf("Unary", {"op", "rhs"})
ast.unspecialized.expr:leaf("Binary", {"op", "lhs", "rhs"})
ast.unspecialized.expr:leaf("Deref", {"value"})

ast.unspecialized:leaf("Block", {"stats"})

ast.unspecialized:inner("stat", {"annotations"})
ast.unspecialized.stat:leaf("If", {"cond", "then_block", "elseif_blocks",
                                   "else_block"})
ast.unspecialized.stat:leaf("Elseif", {"cond", "block"})
ast.unspecialized.stat:leaf("While", {"cond", "block"})
ast.unspecialized.stat:leaf("ForNum", {"name", "type_expr", "values", "block"})
ast.unspecialized.stat:leaf("ForList", {"name", "type_expr", "value", "block"})
ast.unspecialized.stat:leaf("Repeat", {"block", "until_cond"})
ast.unspecialized.stat:leaf("MustEpoch", {"block"})
ast.unspecialized.stat:leaf("Block", {"block"})
ast.unspecialized.stat:leaf("Var", {"var_names", "type_exprs", "values"})
ast.unspecialized.stat:leaf("VarUnpack", {"var_names", "fields", "value"})
ast.unspecialized.stat:leaf("Return", {"value"})
ast.unspecialized.stat:leaf("Break")
ast.unspecialized.stat:leaf("Assignment", {"lhs", "rhs"})
ast.unspecialized.stat:leaf("Reduce", {"op", "lhs", "rhs"})
ast.unspecialized.stat:leaf("Expr", {"expr"})
ast.unspecialized.stat:leaf("Escape", {"expr"})
ast.unspecialized.stat:leaf("RawDelete", {"value"})
ast.unspecialized.stat:leaf("Fence", {"kind", "blocking"})
ast.unspecialized.stat:leaf("ParallelizeWith", {"hints", "block"})

ast.unspecialized:inner("top", {"annotations"})
ast.unspecialized.top:leaf("Task", {"name", "params", "return_type_expr",
                                    "effect_exprs", "body"})
ast.unspecialized.top:leaf("TaskParam", {"param_name", "type_expr"})
ast.unspecialized.top:leaf("Fspace", {"name", "params", "fields",
                                      "constraints"})
ast.unspecialized.top:leaf("FspaceParam", {"param_name", "type_expr"})
ast.unspecialized.top:leaf("FspaceField", {"field_name", "type_expr"})
ast.unspecialized.top:leaf("QuoteExpr", {"expr"})
ast.unspecialized.top:leaf("QuoteStat", {"block"})


-- Node Types (Specialized)

ast:inner("specialized", {"span"})

ast.specialized:inner("region")
ast.specialized.region:leaf("Bare", {"symbol"})
ast.specialized.region:leaf("Root", {"symbol", "fields"})
ast.specialized.region:leaf("Field", {"field_name", "fields"})

ast.specialized:leaf("Constraint", {"lhs", "op", "rhs"})

ast.specialized:leaf("Privilege", {"privileges", "regions"})

ast.specialized:leaf("Coherence", {"coherence_modes", "regions"})

ast.specialized:leaf("Flag", {"flags", "regions"})

ast.specialized:leaf("ConditionVariable", {"symbol"})
ast.specialized:leaf("Condition", {"conditions", "variables"})

ast.specialized:inner("expr", {"annotations"})
ast.specialized.expr:leaf("ID", {"value"})
ast.specialized.expr:leaf("FieldAccess", {"value", "field_name"})
ast.specialized.expr:leaf("IndexAccess", {"value", "index"})
ast.specialized.expr:leaf("MethodCall", {"value", "method_name", "args"})
ast.specialized.expr:leaf("Call", {"fn", "args", "conditions"})
ast.specialized.expr:leaf("Cast", {"fn", "args"})
ast.specialized.expr:leaf("Ctor", {"fields", "named"})
ast.specialized.expr:leaf("CtorListField", {"value"})
ast.specialized.expr:leaf("CtorRecField", {"name", "value"})
ast.specialized.expr:leaf("Constant", {"value", "expr_type"})
ast.specialized.expr:leaf("RawContext")
ast.specialized.expr:leaf("RawFields", {"region"})
ast.specialized.expr:leaf("RawPhysical", {"region"})
ast.specialized.expr:leaf("RawRuntime")
ast.specialized.expr:leaf("RawValue", {"value"})
ast.specialized.expr:leaf("Isnull", {"pointer"})
ast.specialized.expr:leaf("New", {"pointer_type", "region", "extent"})
ast.specialized.expr:leaf("Null", {"pointer_type"})
ast.specialized.expr:leaf("DynamicCast", {"value", "expr_type"})
ast.specialized.expr:leaf("StaticCast", {"value", "expr_type"})
ast.specialized.expr:leaf("UnsafeCast", {"value", "expr_type"})
ast.specialized.expr:leaf("Ispace", {"index_type", "extent", "start"})
ast.specialized.expr:leaf("Region", {"ispace", "fspace_type"})
ast.specialized.expr:leaf("Partition", {"disjointness", "region", "coloring",
                                        "colors"})
ast.specialized.expr:leaf("PartitionEqual", {"region", "colors"})
ast.specialized.expr:leaf("PartitionByField", {"region", "colors"})
ast.specialized.expr:leaf("Image", {"parent", "partition", "region"})
ast.specialized.expr:leaf("Preimage", {"parent", "partition", "region"})
ast.specialized.expr:leaf("CrossProduct", {"args"})
ast.specialized.expr:leaf("CrossProductArray", {"lhs", "disjointness", "colorings"})
ast.specialized.expr:leaf("ListSlicePartition", {"partition", "indices"})
ast.specialized.expr:leaf("ListDuplicatePartition", {"partition", "indices"})
ast.specialized.expr:leaf("ListCrossProduct", {"lhs", "rhs", "shallow"})
ast.specialized.expr:leaf("ListCrossProductComplete", {"lhs", "product"})
ast.specialized.expr:leaf("ListPhaseBarriers", {"product"})
ast.specialized.expr:leaf("ListInvert", {"rhs", "product", "barriers"})
ast.specialized.expr:leaf("ListRange", {"start", "stop"})
ast.specialized.expr:leaf("ListIspace", {"ispace"})
ast.specialized.expr:leaf("ListFromElement", {"list", "value"})
ast.specialized.expr:leaf("PhaseBarrier", {"value"})
ast.specialized.expr:leaf("DynamicCollective", {"value_type", "op", "arrivals"})
ast.specialized.expr:leaf("DynamicCollectiveGetResult", {"value"})
ast.specialized.expr:leaf("Advance", {"value"})
ast.specialized.expr:leaf("Adjust", {"barrier", "value"})
ast.specialized.expr:leaf("Arrive", {"barrier", "value"})
ast.specialized.expr:leaf("Await", {"barrier"})
ast.specialized.expr:leaf("Copy", {"src", "dst", "op", "conditions"})
ast.specialized.expr:leaf("Fill", {"dst", "value", "conditions"})
ast.specialized.expr:leaf("Acquire", {"region", "conditions"})
ast.specialized.expr:leaf("Release", {"region", "conditions"})
ast.specialized.expr:leaf("AttachHDF5", {"region", "filename", "mode"})
ast.specialized.expr:leaf("DetachHDF5", {"region"})
ast.specialized.expr:leaf("AllocateScratchFields", {"region"})
ast.specialized.expr:leaf("WithScratchFields", {"region", "field_ids"})
ast.specialized.expr:leaf("RegionRoot", {"region", "fields"})
ast.specialized.expr:leaf("Condition", {"conditions", "values"})
ast.specialized.expr:leaf("Function", {"value"})
ast.specialized.expr:leaf("Unary", {"op", "rhs"})
ast.specialized.expr:leaf("Binary", {"op", "lhs", "rhs"})
ast.specialized.expr:leaf("Deref", {"value"})
ast.specialized.expr:leaf("LuaTable", {"value"})

ast.specialized:leaf("Block", {"stats"})

ast.specialized:inner("stat", {"annotations"})
ast.specialized.stat:leaf("If", {"cond", "then_block", "elseif_blocks",
                                 "else_block"})
ast.specialized.stat:leaf("Elseif", {"cond", "block"})
ast.specialized.stat:leaf("While", {"cond", "block"})
ast.specialized.stat:leaf("ForNum", {"symbol", "values", "block"})
ast.specialized.stat:leaf("ForList", {"symbol", "value", "block"})
ast.specialized.stat:leaf("Repeat", {"block", "until_cond"})
ast.specialized.stat:leaf("MustEpoch", {"block"})
ast.specialized.stat:leaf("Block", {"block"})
ast.specialized.stat:leaf("Var", {"symbols", "values"})
ast.specialized.stat:leaf("VarUnpack", {"symbols", "fields", "value"})
ast.specialized.stat:leaf("Return", {"value"})
ast.specialized.stat:leaf("Break")
ast.specialized.stat:leaf("Assignment", {"lhs", "rhs"})
ast.specialized.stat:leaf("Reduce", {"op", "lhs", "rhs"})
ast.specialized.stat:leaf("Expr", {"expr"})
ast.specialized.stat:leaf("RawDelete", {"value"})
ast.specialized.stat:leaf("Fence", {"kind", "blocking"})
ast.specialized.stat:leaf("ParallelizeWith", {"hints", "block"})

ast.specialized:inner("top", {"annotations"})
ast.specialized.top:leaf("Task", {"name", "params", "return_type",
                                  "privileges", "coherence_modes", "flags",
                                  "conditions", "constraints", "body",
                                  "prototype"})
ast.specialized.top:leaf("TaskParam", {"symbol", "future"})
ast.specialized.top:leaf("Fspace", {"name", "fspace", "constraints"})
ast.specialized.top:leaf("QuoteExpr", {"expr"})
ast.specialized.top:leaf("QuoteStat", {"block"})


-- Node Types (Typed)

ast.typed = ast:inner("typed", {"span"})

ast.typed:inner("expr", {"annotations", "expr_type"})
ast.typed.expr:leaf("Internal", {"value"}) -- internal use only

ast.typed.expr:leaf("ID", {"value"})
ast.typed.expr:leaf("FieldAccess", {"value", "field_name"})
ast.typed.expr:leaf("IndexAccess", {"value", "index"})
ast.typed.expr:leaf("MethodCall", {"value", "method_name", "args"})
ast.typed.expr:leaf("Call", {"fn", "args", "conditions", "replicable"})
ast.typed.expr:leaf("Cast", {"fn", "arg"})
ast.typed.expr:leaf("Ctor", {"fields", "named"})
ast.typed.expr:leaf("CtorListField", {"value"})
ast.typed.expr:leaf("CtorRecField", {"name", "value"})
ast.typed.expr:leaf("RawContext")
ast.typed.expr:leaf("RawFields", {"region", "fields"})
ast.typed.expr:leaf("RawPhysical", {"region", "fields"})
ast.typed.expr:leaf("RawRuntime")
ast.typed.expr:leaf("RawValue", {"value"})
ast.typed.expr:leaf("Isnull", {"pointer"})
ast.typed.expr:leaf("Null", {"pointer_type"})
ast.typed.expr:leaf("DynamicCast", {"value"})
ast.typed.expr:leaf("StaticCast", {"value", "parent_region_map"})
ast.typed.expr:leaf("UnsafeCast", {"value"})
ast.typed.expr:leaf("Ispace", {"index_type", "extent", "start"})
ast.typed.expr:leaf("Region", {"ispace", "fspace_type"})
ast.typed.expr:leaf("Partition", {"disjointness", "region", "coloring",
                                  "colors"})
ast.typed.expr:leaf("PartitionEqual", {"region", "colors"})
ast.typed.expr:leaf("PartitionByField", {"region", "colors"})
ast.typed.expr:leaf("Image", {"parent", "partition", "region"})
ast.typed.expr:leaf("ImageByTask", {"parent", "partition", "task"})
ast.typed.expr:leaf("Preimage", {"parent", "partition", "region"})
ast.typed.expr:leaf("CrossProduct", {"args"})
ast.typed.expr:leaf("CrossProductArray", {"lhs", "disjointness", "colorings"})
ast.typed.expr:leaf("ListSlicePartition", {"partition", "indices"})
ast.typed.expr:leaf("ListDuplicatePartition", {"partition", "indices"})
ast.typed.expr:leaf("ListSliceCrossProduct", {"product", "indices"})
ast.typed.expr:leaf("ListCrossProduct", {"lhs", "rhs", "shallow"})
ast.typed.expr:leaf("ListCrossProductComplete", {"lhs", "product"})
ast.typed.expr:leaf("ListPhaseBarriers", {"product"})
ast.typed.expr:leaf("ListInvert", {"rhs", "product", "barriers"})
ast.typed.expr:leaf("ListRange", {"start", "stop"})
ast.typed.expr:leaf("ListIspace", {"ispace"})
ast.typed.expr:leaf("ListFromElement", {"list", "value"})
ast.typed.expr:leaf("PhaseBarrier", {"value"})
ast.typed.expr:leaf("DynamicCollective", {"value_type", "op", "arrivals"})
ast.typed.expr:leaf("DynamicCollectiveGetResult", {"value"})
ast.typed.expr:leaf("Advance", {"value"})
ast.typed.expr:leaf("Adjust", {"barrier", "value"})
ast.typed.expr:leaf("Arrive", {"barrier", "value"})
ast.typed.expr:leaf("Await", {"barrier"})
ast.typed.expr:leaf("Copy", {"src", "dst", "op", "conditions"})
ast.typed.expr:leaf("Fill", {"dst", "value", "conditions"})
ast.typed.expr:leaf("Acquire", {"region", "conditions"})
ast.typed.expr:leaf("Release", {"region", "conditions"})
ast.typed.expr:leaf("AttachHDF5", {"region", "filename", "mode"})
ast.typed.expr:leaf("DetachHDF5", {"region"})
ast.typed.expr:leaf("AllocateScratchFields", {"region"})
ast.typed.expr:leaf("WithScratchFields", {"region", "field_ids"})
ast.typed.expr:leaf("RegionRoot", {"region", "fields"})
ast.typed.expr:leaf("Condition", {"conditions", "value"})
ast.typed.expr:leaf("Constant", {"value"})
ast.typed.expr:leaf("Function", {"value"})
ast.typed.expr:leaf("Unary", {"op", "rhs"})
ast.typed.expr:leaf("Binary", {"op", "lhs", "rhs"})
ast.typed.expr:leaf("Deref", {"value"})
ast.typed.expr:leaf("Future", {"value"})
ast.typed.expr:leaf("FutureGetResult", {"value"})
ast.typed.expr:leaf("ParallelizerConstraint", {"lhs", "op", "rhs"})

ast.typed:leaf("Block", {"stats"})

ast.typed:inner("stat", {"annotations"})
ast.typed.stat:leaf("Internal", {"actions"}) -- internal use only
ast.typed.stat:leaf("If", {"cond", "then_block", "elseif_blocks", "else_block"})
ast.typed.stat:leaf("Elseif", {"cond", "block"})
ast.typed.stat:leaf("While", {"cond", "block"})
ast.typed.stat:leaf("ForNum", {"symbol", "values", "block"})
ast.typed.stat:leaf("ForNumVectorized", {"symbol", "values", "block",
                                         "orig_block", "vector_width"})
ast.typed.stat:leaf("ForList", {"symbol", "value", "block"})
ast.typed.stat:leaf("ForListVectorized", {"symbol", "value", "block",
                                          "orig_block", "vector_width"})
ast.typed.stat:leaf("Repeat", {"block", "until_cond"})
ast.typed.stat:leaf("MustEpoch", {"block"})
ast.typed.stat:leaf("Block", {"block"})
ast.typed.stat:leaf("IndexLaunchNum", {"symbol", "values", "preamble", "call",
                                       "reduce_lhs", "reduce_op",
                                       "args_provably"})
ast.typed.stat:leaf("IndexLaunchList", {"symbol", "value", "preamble", "call",
                                        "reduce_lhs", "reduce_op",
                                        "args_provably"})
ast:leaf("IndexLaunchArgsProvably", {"invariant", "projectable"})
ast.typed.stat:leaf("Var", {"symbol", "type", "value"})
ast.typed.stat:leaf("VarUnpack", {"symbols", "fields", "field_types", "value"})
ast.typed.stat:leaf("Return", {"value"})
ast.typed.stat:leaf("Break")
ast.typed.stat:leaf("Assignment", {"lhs", "rhs"})
ast.typed.stat:leaf("Reduce", {"op", "lhs", "rhs"})
ast.typed.stat:leaf("Expr", {"expr"})
ast.typed.stat:leaf("RawDelete", {"value"})
ast.typed.stat:leaf("Fence", {"kind", "blocking"})
ast.typed.stat:leaf("ParallelizeWith", {"hints", "block"})
ast.typed.stat:leaf("BeginTrace", {"trace_id"})
ast.typed.stat:leaf("EndTrace", {"trace_id"})
ast.typed.stat:leaf("MapRegions", {"region_types"})
ast.typed.stat:leaf("UnmapRegions", {"region_types"})

ast:leaf("TaskConfigOptions", {"leaf", "inner", "idempotent"})

ast.typed:inner("top", {"annotations"})
ast.typed.top:leaf("Fspace", {"name", "fspace"})
ast.typed.top:leaf("Task", {"name", "params", "return_type", "privileges",
                             "coherence_modes", "flags", "conditions",
                             "constraints", "body", "config_options",
                             "region_divergence", "prototype"})
ast.typed.top:leaf("TaskParam", {"symbol", "param_type", "future"})

return ast
