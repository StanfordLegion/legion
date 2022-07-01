-- Copyright 2022 Stanford University
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

-- Regent Index Launch Optimizer
--
-- Attempts to determine which loops can be transformed into index
-- space task launches.

local affine_helper = require("regent/affine_helper")
local ast = require("regent/ast")
local util = require("regent/ast_util")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")

local skip_interference_check = std.config["override-demand-index-launch"]
local emit_dynamic_check = std.config["index-launch-dynamic"]

local context = {}

local function get_source_location(node)
  assert(node.span.source and node.span.start.line)
  return tostring(node.span.source) .. ":" .. tostring(node.span.start.line)
end

local function strip_projection(node)
  if node:is(ast.typed.expr.Projection) then
    node = node.region
  end
  return node
end

local function replace_projection(node, replacement)
  if node:is(ast.typed.expr.Projection) then
    return node { region = replacement }
  end
  return replacement
end

local function get_base_partition_symbol(node)
  node = util.get_base_indexed_node(strip_projection(node))
  if std.is_cross_product(std.as_read(node.expr_type)) then
    return std.as_read(node.expr_type).partition_symbols[1]
  end
  return node.value
end

local function get_partition_dim(expr)
  return std.as_read(expr.expr_type):colors().dim
end

function context:__index (field)
  local value = context [field]
  if value ~= nil then
    return value
  end
  error ("context has no field '" .. field .. "' (in lookup)", 2)
end

function context:__newindex (field, value)
  error ("context has no field '" .. field .. "' (in assignment)", 2)
end

function context:new_local_scope()
  local cx = {
    constraints = self.constraints,
    loop_index = false,
    loop_variables = data.newmap(),
    free_variables = data.newmap(),
  }
  return setmetatable(cx, context)
end

function context:new_task_scope(constraints)
  local cx = {
    constraints = constraints,
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

function context:set_loop_index(loop_index)
  assert(not self.loop_index)
  self.loop_index = loop_index
end

function context:add_loop_variable(loop_variable)
  assert(self.loop_variables)
  self.loop_variables[loop_variable] = true
end

function context:add_free_variable(symbol)
  assert(self.free_variables)
  self.free_variables[symbol] = true
end

function context:is_free_variable(variable)
  assert(self.free_variables)
  return self.free_variables[variable]
end

function context:is_loop_variable(variable)
  assert(self.loop_variables)
  return self.loop_variables[variable]
end

function context:is_loop_index(variable)
  assert(self.loop_index)
  return self.loop_index == variable
end

local result = ast.make_factory("result")
result:inner("node")
result.node:leaf("Variant", {"coefficient"}):set_memoize()
result.node:leaf("MultDim", {"coeff_matrix"})
result.node:leaf("Constant", {"value"}):set_memoize()
result.node:leaf("Invariant", {}):set_memoize()

data.matrix = {}
setmetatable(data.matrix, {__index = data.vector })
data.matrix.__index = data.matrix

function data.is_matrix(x)
  return getmetatable(x) == data.matrix
end

function data.matrix.__add(a, b)
  assert(data.is_matrix(a) and data.is_matrix(b))

  local result = data.newmatrix()
  for i, row in ipairs(a) do
    local result_row = data.newvector()
    for j, elem in ipairs(row) do
      result_row:insert(elem + (b[i][j] or 0))
    end
    result:insert(result_row)
  end
  return result
end

function data.matrix.__mul(a, b)
  if data.is_matrix(a) then
    assert(type(b) == "number")
    if b == 1 then return a end

    local result = data.newmatrix()
    for i, row in ipairs(a) do
      result:insert(b * row)
    end
    return result
  elseif data.is_matrix(b) then
    return data.matrix.__mul(b,a)
  end
  assert(false) -- At least one should have been a matrix
end

function data.newmatrix(rows, cols)
  if rows then
    cols = cols or rows
    local mat = data.newvector()
    for i = 1, rows do
      local row = data.newvector()
      for j = 1, cols do
        row:insert(i == j and 1 or 0)
      end
      mat:insert(row)
    end
    return setmetatable(mat, data.matrix)
  end
  return setmetatable( {}, data.matrix)
end

local function get_privileges_before_projection(expr, privileges, coherence_modes)
  if expr:is(ast.typed.expr.Projection) then
    local function map_domain(map, tbl)
      local result = data.newmap()
      for k, v in tbl:items() do
        assert(map[k] ~= nil)
        result[map[k]] = v
      end
      return result
    end
    local inv_field_mapping = data.dict(expr.field_mapping:map(
      function(pair) return { pair[2], pair[1] } end))
    privileges = map_domain(inv_field_mapping, privileges)
    coherence_modes = map_domain(inv_field_mapping, coherence_modes)
  end
  return privileges, coherence_modes
end

local function check_privilege_noninterference(cx, task, arg,
                                         other_arg, mapping)
  local region_type = std.as_read(arg.expr_type)
  local other_region_type = std.as_read(other_arg.expr_type)
  local param_region_type = mapping[arg]
  local other_param_region_type = mapping[other_arg]
  assert(param_region_type and other_param_region_type)

  local privileges_by_field_path, coherence_modes_by_field_path =
    get_privileges_before_projection(arg,
      std.group_task_privileges_by_field_path(
        std.find_task_privileges(param_region_type, task)))
  local other_privileges_by_field_path, other_coherence_modes_by_field_path =
    get_privileges_before_projection(other_arg,
      std.group_task_privileges_by_field_path(
        std.find_task_privileges(other_param_region_type, task)))

  for field_path, privilege in privileges_by_field_path:items() do
    local other_privilege = other_privileges_by_field_path[field_path]

    if not (
        not privilege or privilege == "none" or
        not other_privilege or other_privilege == "none" or
        (privilege == "reads" and other_privilege == "reads") or
        (std.is_reduction_op(privilege) and privilege == other_privilege) or
        (coherence_modes_by_field_path[field_path] == "simultaneous" and
         other_coherence_modes_by_field_path[field_path] == "simultaneous"))
    then
      return false
    end
  end
  return true
end

local function strip_casts(node)
  if node:is(ast.typed.expr.Cast) then
    return node.arg
  end
  return node
end

local function analyze_noninterference_previous(
    cx, task, arg, regions_previously_used, mapping)
  local region_type = std.as_read(arg.expr_type)
  if region_type:is_projected() then
    region_type = region_type:get_projection_source()
  end
  local args_interfering = terralib.newlist()
  for i, other_arg in ipairs(regions_previously_used) do
    local other_region_type = std.as_read(other_arg.expr_type)
    if other_region_type:is_projected() then
      other_region_type = other_region_type:get_projection_source()
    end
    local constraint = std.constraint(
      region_type,
      other_region_type,
      std.disjointness)
    local exclude_variables = { [cx.loop_index] = true }

    if not (
        not std.type_maybe_eq(region_type.fspace_type, other_region_type.fspace_type) or
        std.check_constraint(cx, constraint, exclude_variables) or
        check_privilege_noninterference(cx, task, arg, other_arg, mapping))
        -- Index non-interference is handled at the type checker level
        -- and is captured in the constraints.
    then
      args_interfering:insert(i)
    end
  end

  if next(args_interfering) == nil then
    return true
  end
  return false, args_interfering
end

local function add_exprs(lhs, rhs, sign)
  if not (lhs and rhs) then
    return false
  end

  if lhs:is(result.node.MultDim) or rhs:is(result.node.MultDim) then
    if lhs:is(result.node.MultDim) and rhs:is(result.node.MultDim) then
      return result.node.MultDim {
        coeff_matrix = lhs.coeff_matrix + sign * rhs.coeff_matrix,
      }

    elseif lhs:is(result.node.Variant) then
      return result.node.MultDim {
        coeff_matrix = lhs.coefficient * data.newmatrix(#rhs.coeff_matrix[1], #rhs.coeff_matrix) + sign * rhs.coeff_matrix,
      }

    elseif rhs:is(result.node.Variant) then
      return result.node.MultDim {
        coeff_matrix = lhs.coeff_matrix + sign * rhs.coefficient * data.newmatrix(#lhs.coeff_matrix[1], #lhs.coeff_matrix),
      }

    else
      -- Adding a const or invariant, return MultDim
      return lhs:is(result.node.MultDim) and lhs or rhs
    end

  elseif lhs:is(result.node.Variant) or rhs:is(result.node.Variant) then
    local coeff = (lhs:is(result.node.Variant) and lhs.coefficient or 0) +
      (rhs:is(result.node.Variant) and sign * rhs.coefficient or 0)
    return result.node.Variant {
      coefficient = coeff,
    }

  elseif lhs:is(result.node.Constant) and rhs:is(result.node.Constant) then
    return result.node.Constant {
      value = lhs.value + sign * rhs.value
    }

  else
    return result.node.Invariant {}
  end
end

local function mult_exprs(lhs, rhs)
  if not (lhs and rhs) then
    return false
  end

  if lhs:is(result.node.Constant) and rhs:is(result.node.Constant) then
    return result.node.Constant {
      value = lhs.value * rhs.value
    }

  elseif lhs:is(result.node.Constant) or rhs:is(result.node.Constant) then
    if rhs:is(result.node.Variant) then
      return result.node.Variant {
        coefficient = lhs.value * rhs.coefficient,
      }

    elseif rhs:is(result.node.MultDim) then
      return result.node.MultDim {
        coeff_matrix = lhs.value * rhs.coeff_matrix,
      }

    elseif rhs:is(result.node.Invariant) then
      return result.node.Invariant {}
    end

    return mult_exprs(rhs, lhs)

  elseif lhs:is(result.node.Invariant) and rhs:is(result.node.Invariant) then
    return result.node.Invariant {}
  end

  -- all other combinations invalid
  return false
end

function analyze_expr_noninterference_self(expression, cx, loop_vars, report_fail, field_name)
  local expr = strip_casts(expression)

  if expr:is(ast.typed.expr.ID) then
    if cx:is_loop_index(expr.value) then
      return result.node.Variant {
        coefficient = 1,
      }

    elseif cx:is_loop_variable(expr.value) then
      for _, loop_var in ipairs(loop_vars) do
        if loop_var.symbol == expr.value then
          return analyze_expr_noninterference_self(loop_var.value, cx, loop_vars, report_fail, field_name)
        end
      end
      assert(false) -- loop_variable should have been found

    else
      return result.node.Invariant {}
    end

  elseif expr:is(ast.typed.expr.Constant) then
    return result.node.Constant {
      value = expr.value,
    }

  elseif expr:is(ast.typed.expr.FieldAccess) then
    local id = expr.value
    if cx:is_loop_index(id.value) and field_name == expr.field_name then
      return result.node.Variant {
        coefficient = 1,
      }
    else
      return result.node.Invariant {}
    end

  elseif expr:is(ast.typed.expr.Binary) then
    local lhs = analyze_expr_noninterference_self(expr.lhs, cx, loop_vars, report_fail, field_name)
    local rhs =  analyze_expr_noninterference_self(expr.rhs, cx, loop_vars, report_fail, field_name)

    if expr.op == "+" then
      return add_exprs(lhs, rhs, 1)

    elseif expr.op == "-" then
      return add_exprs(lhs, rhs, -1)

    elseif expr.op == "*" then
      return mult_exprs(lhs, rhs)

    -- TODO: add mod operator check
    else
      return false
    end

  elseif expr:is(ast.typed.expr.Ctor) then
    local loop_index_type = cx.loop_index:gettype()

    local coeff_mat = data.newmatrix()
    for i, ctor_field in ipairs(expr.fields) do
      local result_row = data.newvector()
      if loop_index_type.fields then
        for j, loop_index_field in ipairs(loop_index_type.fields) do
          local res = analyze_expr_noninterference_self(ctor_field.value, cx, loop_vars, report_fail, loop_index_field)
          if not res then
            result_row:insert(false)
            break
          end
          result_row:insert(res and (res:is(result.node.Variant) and res.coefficient or 0))
        end
      else
        local res = analyze_expr_noninterference_self(ctor_field.value, cx, loop_vars, report_fail, field_name)
        result_row:insert(res and (res:is(result.node.Variant) and res.coefficient or 0))
      end
      coeff_mat:insert(result_row)
    end
    return result.node.MultDim {
      coeff_matrix = coeff_mat,
    }
  end

  return false
end

-- TODO: replace this test with proper solver (prove injective transformation):
-- Ex: solve Kernel of Matrix and assert only solution is zero vector
local function matrix_is_noninterfering(matrix, cx)
  local stat = terralib.newlist()
  if cx.loop_index:gettype().fields then
    for _, _ in ipairs(cx.loop_index:gettype().fields) do
      stat:insert(false)
    end
  else
    stat:insert(false)
  end

  local stat_avail = terralib.newlist()
  for i = 1, #matrix do
    stat_avail:insert(true)
  end

  local field = 1
  local row = 1

  while field > 0 do
    if row > #matrix then
      field = field -1
      if field < 1 then
        return false
      end
      row = stat[field] + 1
      stat_avail[stat[field]] = true
    elseif stat_avail[row] and matrix[row][field] and matrix[row][field] ~= 0 then
      stat[field] = row
      stat_avail[row] = false
      if field == #stat then
        return true
      end
      field = field + 1
      row = 1
    else
      row = row + 1
    end
  end

  return false
end

local function analyze_index_noninterference_self(expr, cx, loop_vars, report_fail, field_name)
  local res = analyze_expr_noninterference_self(expr, cx, loop_vars, report_fail, field_name)
  if not res then
    return false
  elseif res:is(result.node.Variant) then
    return res.coefficient ~= 0
  elseif res:is(result.node.MultDim) then
    return matrix_is_noninterfering(res.coeff_matrix, cx)
  end
  return false
end

local function analyze_noninterference_self(
    cx, task, arg, partition_type, mapping, loop_vars)
  local is_disjoint = partition_type and partition_type:is_disjoint()
  if is_disjoint then
    local _, index = util.get_base_indexed_node(strip_projection(arg))

    if analyze_index_noninterference_self(index, cx, loop_vars) then
      return true, false
    end
  end

  local param_region_type = mapping[arg]
  assert(param_region_type)
  local privileges, privilege_field_paths, privilege_field_types,
        privilege_coherence_modes, privilege_flags = std.find_task_privileges(
    param_region_type, task)
  for i, privilege in ipairs(privileges) do
    local field_paths = privilege_field_paths[i]
    local coherence = privilege_coherence_modes[i]
    if not (
        privilege == "none" or
        privilege == "reads" or
        std.is_reduction_op(privilege) or
        coherence == "simultaneous" or
        #field_paths == 0)
    then
       if not is_disjoint then
         return false, false
       else
         -- The only case where we emit the dynamic check
         return false, true
       end
    end
  end
  return true, false
end

local function always_true(cx, node)
  return true
end

local function always_false(cx, node)
  return false
end

local function unreachable(cx, node)
  assert(false, "unreachable")
end

local node_is_side_effect_free = {
  -- Expressions:
  [ast.typed.expr.IndexAccess] = function(cx, node)
    return not std.is_ref(node.expr_type)
  end,

  [ast.typed.expr.Call] = function(cx, node)
    return not std.is_task(node.fn.value)
  end,

  [ast.typed.expr.RawContext]                 = always_false,
  [ast.typed.expr.RawPhysical]                = always_false,
  [ast.typed.expr.RawRuntime]                 = always_false,
  [ast.typed.expr.Ispace]                     = always_false,
  [ast.typed.expr.Region]                     = always_false,
  [ast.typed.expr.Partition]                  = always_false,
  [ast.typed.expr.PartitionEqual]             = always_false,
  [ast.typed.expr.PartitionByField]           = always_false,
  [ast.typed.expr.PartitionByRestriction]     = always_false,
  [ast.typed.expr.Image]                      = always_false,
  [ast.typed.expr.ImageByTask]                = always_false,
  [ast.typed.expr.Preimage]                   = always_false,
  [ast.typed.expr.CrossProduct]               = always_false,
  [ast.typed.expr.CrossProductArray]          = always_false,
  [ast.typed.expr.ListSlicePartition]         = always_false,
  [ast.typed.expr.ListDuplicatePartition]     = always_false,
  [ast.typed.expr.ListSliceCrossProduct]      = always_false,
  [ast.typed.expr.ListCrossProduct]           = always_false,
  [ast.typed.expr.ListCrossProductComplete]   = always_false,
  [ast.typed.expr.ListPhaseBarriers]          = always_false,
  [ast.typed.expr.PhaseBarrier]               = always_false,
  [ast.typed.expr.DynamicCollective]          = always_false,
  [ast.typed.expr.Adjust]                     = always_false,
  [ast.typed.expr.Arrive]                     = always_false,
  [ast.typed.expr.Await]                      = always_false,
  [ast.typed.expr.Copy]                       = always_false,
  [ast.typed.expr.Fill]                       = always_false,
  [ast.typed.expr.Acquire]                    = always_false,
  [ast.typed.expr.Release]                    = always_false,
  [ast.typed.expr.AttachHDF5]                 = always_false,
  [ast.typed.expr.DetachHDF5]                 = always_false,
  [ast.typed.expr.AllocateScratchFields]      = always_false,
  [ast.typed.expr.Condition]                  = always_false,
  [ast.typed.expr.Deref]                      = always_false,
  [ast.typed.expr.ImportIspace]               = always_false,
  [ast.typed.expr.ImportRegion]               = always_false,
  [ast.typed.expr.ImportPartition]            = always_false,
  [ast.typed.expr.ImportCrossProduct]         = always_false,

  [ast.typed.expr.ID]                         = always_true,
  [ast.typed.expr.Constant]                   = always_true,
  [ast.typed.expr.Global]                     = always_true,
  [ast.typed.expr.Function]                   = always_true,
  [ast.typed.expr.FieldAccess]                = always_true,
  [ast.typed.expr.MethodCall]                 = always_true,
  [ast.typed.expr.Cast]                       = always_true,
  [ast.typed.expr.Ctor]                       = always_true,
  [ast.typed.expr.CtorListField]              = always_true,
  [ast.typed.expr.CtorRecField]               = always_true,
  [ast.typed.expr.RawFields]                  = always_true,
  [ast.typed.expr.RawFuture]                  = always_true,
  [ast.typed.expr.RawTask]                    = always_true,
  [ast.typed.expr.RawValue]                   = always_true,
  [ast.typed.expr.Isnull]                     = always_true,
  [ast.typed.expr.Null]                       = always_true,
  [ast.typed.expr.DynamicCast]                = always_true,
  [ast.typed.expr.StaticCast]                 = always_true,
  [ast.typed.expr.UnsafeCast]                 = always_true,
  [ast.typed.expr.ListInvert]                 = always_true,
  [ast.typed.expr.ListRange]                  = always_true,
  [ast.typed.expr.ListIspace]                 = always_true,
  [ast.typed.expr.ListFromElement]            = always_true,
  [ast.typed.expr.DynamicCollectiveGetResult] = always_true,
  [ast.typed.expr.Advance]                    = always_true,
  [ast.typed.expr.WithScratchFields]          = always_true,
  [ast.typed.expr.RegionRoot]                 = always_true,
  [ast.typed.expr.Unary]                      = always_true,
  [ast.typed.expr.Binary]                     = always_true,
  [ast.typed.expr.AddressOf]                  = always_true,
  [ast.typed.expr.ParallelizerConstraint]     = always_true,
  [ast.typed.expr.Projection]                 = always_true,

  [ast.typed.expr.Future]                     = unreachable,
  [ast.typed.expr.FutureGetResult]            = unreachable,
  [ast.typed.expr.Internal]                   = unreachable,

  -- Statements:
  [ast.typed.stat.Var]                        = always_true,
}

local analyze_is_side_effect_free_node = ast.make_single_dispatch(
  node_is_side_effect_free,
  {ast.typed.expr})

local function analyze_is_side_effect_free(cx, node)
  return ast.mapreduce_expr_postorder(
    analyze_is_side_effect_free_node(cx),
    data.all,
    node, true)
end

local node_is_loop_invariant = {
  -- Expressions:
  [ast.typed.expr.ID] = function(cx, node)
    return not cx:is_loop_variable(node.value)
  end,

  [ast.typed.expr.IndexAccess] = function(cx, node)
    return not std.is_ref(node.expr_type)
  end,

  [ast.typed.expr.Call]                       = always_false,
  [ast.typed.expr.Ispace]                     = always_false,
  [ast.typed.expr.Region]                     = always_false,
  [ast.typed.expr.Partition]                  = always_false,
  [ast.typed.expr.PartitionEqual]             = always_false,
  [ast.typed.expr.PartitionByField]           = always_false,
  [ast.typed.expr.PartitionByRestriction]     = always_false,
  [ast.typed.expr.Image]                      = always_false,
  [ast.typed.expr.ImageByTask]                = always_false,
  [ast.typed.expr.Preimage]                   = always_false,
  [ast.typed.expr.CrossProduct]               = always_false,
  [ast.typed.expr.CrossProductArray]          = always_false,
  [ast.typed.expr.ListSlicePartition]         = always_false,
  [ast.typed.expr.ListDuplicatePartition]     = always_false,
  [ast.typed.expr.ListSliceCrossProduct]      = always_false,
  [ast.typed.expr.ListCrossProduct]           = always_false,
  [ast.typed.expr.ListCrossProductComplete]   = always_false,
  [ast.typed.expr.ListPhaseBarriers]          = always_false,
  [ast.typed.expr.PhaseBarrier]               = always_false,
  [ast.typed.expr.DynamicCollective]          = always_false,
  [ast.typed.expr.DynamicCollectiveGetResult] = always_false,
  [ast.typed.expr.Adjust]                     = always_false,
  [ast.typed.expr.Arrive]                     = always_false,
  [ast.typed.expr.Await]                      = always_false,
  [ast.typed.expr.Copy]                       = always_false,
  [ast.typed.expr.Fill]                       = always_false,
  [ast.typed.expr.Acquire]                    = always_false,
  [ast.typed.expr.Release]                    = always_false,
  [ast.typed.expr.AttachHDF5]                 = always_false,
  [ast.typed.expr.DetachHDF5]                 = always_false,
  [ast.typed.expr.AllocateScratchFields]      = always_false,
  [ast.typed.expr.Condition]                  = always_false,
  [ast.typed.expr.Deref]                      = always_false,
  [ast.typed.expr.ImportIspace]               = always_false,
  [ast.typed.expr.ImportRegion]               = always_false,
  [ast.typed.expr.ImportPartition]            = always_false,
  [ast.typed.expr.ImportCrossProduct]         = always_false,
  [ast.typed.expr.Projection]                 = always_false,

  [ast.typed.expr.Constant]                   = always_true,
  [ast.typed.expr.Global]                     = always_true,
  [ast.typed.expr.Function]                   = always_true,
  [ast.typed.expr.FieldAccess]                = always_true,
  [ast.typed.expr.MethodCall]                 = always_true,
  [ast.typed.expr.Cast]                       = always_true,
  [ast.typed.expr.Ctor]                       = always_true,
  [ast.typed.expr.CtorListField]              = always_true,
  [ast.typed.expr.CtorRecField]               = always_true,
  [ast.typed.expr.RawContext]                 = always_true,
  [ast.typed.expr.RawFields]                  = always_true,
  [ast.typed.expr.RawFuture]                  = always_true,
  [ast.typed.expr.RawPhysical]                = always_true,
  [ast.typed.expr.RawRuntime]                 = always_true,
  [ast.typed.expr.RawTask]                    = always_true,
  [ast.typed.expr.RawValue]                   = always_true,
  [ast.typed.expr.Isnull]                     = always_true,
  [ast.typed.expr.Null]                       = always_true,
  [ast.typed.expr.DynamicCast]                = always_true,
  [ast.typed.expr.StaticCast]                 = always_true,
  [ast.typed.expr.UnsafeCast]                 = always_true,
  [ast.typed.expr.ListInvert]                 = always_true,
  [ast.typed.expr.ListRange]                  = always_true,
  [ast.typed.expr.ListIspace]                 = always_true,
  [ast.typed.expr.ListFromElement]            = always_true,
  [ast.typed.expr.Advance]                    = always_true,
  [ast.typed.expr.WithScratchFields]          = always_true,
  [ast.typed.expr.RegionRoot]                 = always_true,
  [ast.typed.expr.Unary]                      = always_true,
  [ast.typed.expr.Binary]                     = always_true,
  [ast.typed.expr.AddressOf]                  = always_true,
  [ast.typed.expr.ParallelizerConstraint]     = always_true,

  [ast.typed.expr.Future]                     = unreachable,
  [ast.typed.expr.FutureGetResult]            = unreachable,
  [ast.typed.expr.Internal]                   = unreachable,
}

local analyze_is_loop_invariant_node = ast.make_single_dispatch(
  node_is_loop_invariant,
  {ast.typed.expr})

local function analyze_is_loop_invariant(cx, node)
  return ast.mapreduce_expr_postorder(
    analyze_is_loop_invariant_node(cx),
    data.all,
    node, true)
end

local function collect_free_variables_node(cx)
  return function(node)
    if node:is(ast.typed.expr.ID) and
      not std.is_region(node.value:gettype()) and
      not std.is_partition(node.value:gettype()) and
      not cx:is_loop_variable(node.value) and
      not cx:is_free_variable(node.value)
    then
      cx:add_free_variable(node.value)
    end
  end
end

local function collect_free_variables(cx, node)
  return ast.mapreduce_expr_postorder(
    collect_free_variables_node(cx),
    data.all,
    node, true)
end

local node_is_simple_index_expression = {
  -- Expressions:
  [ast.typed.expr.ID] = function(cx, node)
    return true
  end,

  [ast.typed.expr.FieldAccess] = function(cx, node)
    -- Field access gets desugared in the type checker, just sanity
    -- check here that we're not doing a region access.
    local value_type = std.as_read(node.value.expr_type)
    assert(not std.is_bounded_type(value_type) or std.get_field(value_type.index_type.base_type, node.field_name))
  end,

  [ast.typed.expr.IndexAccess]                = always_false,
  [ast.typed.expr.MethodCall]                 = always_false,
  [ast.typed.expr.Call]                       = always_false,
  [ast.typed.expr.RawContext]                 = always_false,
  [ast.typed.expr.RawFields]                  = always_false,
  [ast.typed.expr.RawFuture]                  = always_false,
  [ast.typed.expr.RawPhysical]                = always_false,
  [ast.typed.expr.RawRuntime]                 = always_false,
  [ast.typed.expr.RawTask]                    = always_false,
  [ast.typed.expr.RawValue]                   = always_false,
  [ast.typed.expr.Isnull]                     = always_false,
  [ast.typed.expr.Null]                       = always_false,
  [ast.typed.expr.DynamicCast]                = always_false,
  [ast.typed.expr.StaticCast]                 = always_false,
  [ast.typed.expr.UnsafeCast]                 = always_false,
  [ast.typed.expr.Ispace]                     = always_false,
  [ast.typed.expr.Region]                     = always_false,
  [ast.typed.expr.Partition]                  = always_false,
  [ast.typed.expr.PartitionEqual]             = always_false,
  [ast.typed.expr.PartitionByField]           = always_false,
  [ast.typed.expr.PartitionByRestriction]     = always_false,
  [ast.typed.expr.Image]                      = always_false,
  [ast.typed.expr.ImageByTask]                = always_false,
  [ast.typed.expr.Preimage]                   = always_false,
  [ast.typed.expr.CrossProduct]               = always_false,
  [ast.typed.expr.CrossProductArray]          = always_false,
  [ast.typed.expr.ListSlicePartition]         = always_false,
  [ast.typed.expr.ListDuplicatePartition]     = always_false,
  [ast.typed.expr.ListSliceCrossProduct]      = always_false,
  [ast.typed.expr.ListCrossProduct]           = always_false,
  [ast.typed.expr.ListCrossProductComplete]   = always_false,
  [ast.typed.expr.ListPhaseBarriers]          = always_false,
  [ast.typed.expr.ListInvert]                 = always_false,
  [ast.typed.expr.ListRange]                  = always_false,
  [ast.typed.expr.ListIspace]                 = always_false,
  [ast.typed.expr.ListFromElement]            = always_false,
  [ast.typed.expr.PhaseBarrier]               = always_false,
  [ast.typed.expr.DynamicCollective]          = always_false,
  [ast.typed.expr.DynamicCollectiveGetResult] = always_false,
  [ast.typed.expr.Advance]                    = always_false,
  [ast.typed.expr.Adjust]                     = always_false,
  [ast.typed.expr.Arrive]                     = always_false,
  [ast.typed.expr.Await]                      = always_false,
  [ast.typed.expr.Copy]                       = always_false,
  [ast.typed.expr.Fill]                       = always_false,
  [ast.typed.expr.Acquire]                    = always_false,
  [ast.typed.expr.Release]                    = always_false,
  [ast.typed.expr.AttachHDF5]                 = always_false,
  [ast.typed.expr.DetachHDF5]                 = always_false,
  [ast.typed.expr.AllocateScratchFields]      = always_false,
  [ast.typed.expr.WithScratchFields]          = always_false,
  [ast.typed.expr.RegionRoot]                 = always_false,
  [ast.typed.expr.Condition]                  = always_false,
  [ast.typed.expr.Deref]                      = always_false,
  [ast.typed.expr.AddressOf]                  = always_false,
  [ast.typed.expr.ParallelizerConstraint]     = always_false,
  [ast.typed.expr.ImportIspace]               = always_false,
  [ast.typed.expr.ImportRegion]               = always_false,
  [ast.typed.expr.ImportPartition]            = always_false,
  [ast.typed.expr.ImportCrossProduct]         = always_false,
  [ast.typed.expr.Projection]                 = always_false,

  [ast.typed.expr.Constant]                   = always_true,
  [ast.typed.expr.Global]                     = always_true,
  [ast.typed.expr.Function]                   = always_true,
  [ast.typed.expr.Cast]                       = always_true,
  [ast.typed.expr.Ctor]                       = always_true,
  [ast.typed.expr.CtorListField]              = always_true,
  [ast.typed.expr.CtorRecField]               = always_true,
  [ast.typed.expr.Unary]                      = always_true,
  [ast.typed.expr.Binary]                     = always_true,

  [ast.typed.expr.Future]                     = unreachable,
  [ast.typed.expr.FutureGetResult]            = unreachable,
  [ast.typed.expr.Internal]                   = unreachable,
}

local analyze_is_simple_index_expression_node = ast.make_single_dispatch(
  node_is_simple_index_expression,
  {ast.typed.expr})

local function analyze_is_simple_index_expression(cx, node)
  return ast.mapreduce_expr_postorder(
    analyze_is_simple_index_expression_node(cx),
    data.all,
    node, true)
end

local function analyze_is_projectable(cx, arg)
  arg = strip_projection(arg)

  -- 1. We can project any index access `p[...]`
  if not arg:is(ast.typed.expr.IndexAccess) then
    return false
  end

  -- 2. As long as `p` is a partition or cross product (otherwise this
  -- is irrelevant, since we wouldn't be producing a region requirement).
  if not (std.is_partition(std.as_read(arg.value.expr_type)) or
       std.is_cross_product(std.as_read(arg.value.expr_type)))
  then
    return false
  end

  -- 3. And as long as `p` is loop-invariant (we have to index from
  -- the same partition or base cross product every time).
  if not analyze_is_loop_invariant(cx, util.get_base_indexed_node(arg)) then
    return false
  end

  -- 4. And as long as the index itself is a simple expression of the
  -- loop index.
  return analyze_is_simple_index_expression(cx, arg.index)
end

local optimize_index_launch = {}
local optimize_index_launches = {}

local function ignore(...) end

local function optimize_loop_body(cx, node, report_pass, report_fail)
  local is_demand_ctl = node.annotations.constant_time_launch:is(ast.annotation.Demand)
  local is_demand = is_demand_ctl or node.annotations.index_launch:is(ast.annotation.Demand)

  local report_fail_ctl = ignore
  if is_demand_ctl then
    report_fail_ctl = report_fail
  end

  if #node.block.stats == 0 then
    report_fail(node, "loop optimization failed: body is empty")
    return
  end

  local loop_cx = cx:new_local_scope()
  loop_cx:set_loop_index(node.symbol)
  loop_cx:add_loop_variable(node.symbol)

  local preamble = terralib.newlist()
  -- vars that need to be defined within projection functor
  local loop_vars = terralib.newlist()
  local call_stat
  for i = 1, #node.block.stats - 1 do
    local stat = node.block.stats[i]
    if not stat:is(ast.typed.stat.Var) then
      report_fail(stat, "loop optimization failed: preamble statement is not a variable")
      return
    end
    if not analyze_is_side_effect_free(loop_cx, stat) then
      if not (i == #node.block.stats - 1 and
              stat.value and
              stat.value:is(ast.typed.expr.Call)) then
        report_fail(stat, "loop optimization failed: preamble statement is not side-effect free")
        return
      else
        call_stat = stat
      end
    end

    if call_stat == nil then
      if stat.value and not analyze_is_loop_invariant(loop_cx, stat.value) then
        loop_cx:add_loop_variable(stat.symbol)
        loop_vars:insert(stat)
      end
      preamble:insert(stat)
    end
  end

  local body = node.block.stats[#node.block.stats]
  local call
  local reduce_lhs, reduce_op = false, false
  if call_stat ~= nil then
    if body:is(ast.typed.stat.Reduce) and
      body.rhs:is(ast.typed.expr.ID) and
      call_stat.symbol == body.rhs.value
    then
      call = call_stat.value
      reduce_lhs = body.lhs
      reduce_op = body.op
    else
      report_fail(call_stat, "loop optimization failed: preamble statement is not side-effect free")
      return
    end
  else
    if body:is(ast.typed.stat.Expr) and
      (body.expr:is(ast.typed.expr.Call) or
       body.expr:is(ast.typed.expr.Fill))
    then
      call = body.expr
    elseif body:is(ast.typed.stat.Reduce) and
      body.rhs:is(ast.typed.expr.Call)
    then
      call = body.rhs
      reduce_lhs = body.lhs
      reduce_op = body.op
    else
      report_fail(body, "loop optimization failed: body is not a function call")
      return
    end
  end

  if call:is(ast.typed.expr.Call) then
    if not std.is_task(call.fn.value) then
      report_fail(call, "loop optimization failed: function is not a task")
      return
    end
    if call.fn.value.is_local then
      report_fail(call, "loop optimization failed: local task cannot be used in an index launch")
      return
    end
  end

  if #call.conditions > 0 then
    report_fail(call, "FIXME: handle analysis of ad-hoc conditions")
    return
  end

  if reduce_lhs then
    local reduce_as_type = std.as_read(call.expr_type)
    if not std.reduction_op_ids[reduce_op][reduce_as_type] then
      report_fail(body, "loop optimization failed: reduction over " .. tostring(reduce_op) .. " " .. tostring(reduce_as_type) .. " not supported")
    end

    if not analyze_is_side_effect_free(loop_cx, reduce_lhs) then
      report_fail(body, "loop optimization failed: reduction is not side-effect free")
    end
  end

  -- Perform static dependence analysis on the loop body to make sure
  -- that the loop is safe to run in parallel. This analysis proceeds
  -- by induction on arguments.
  --
  -- For each argument:
  --
  --  0. Determine whether the argument is provably side-effect
  --     free. (Technically, what this actually does is prove that any
  --     side-effects are non-interfering with the task call itself
  --     and therefore can be split out from the index launch.)
  --
  --  1. Determine whether the argument is provably loop-invariant or
  --     provably "loop-variant" (i.e., a simple function of the loop
  --     index). This is important because index space task launches
  --     are restricted in the forms of region (and index space)
  --     requirements they can accept. All regions (and index spaces)
  --     must be one of:
  --
  --      a. Provably loop-invariant.
  --      b. Provably projectable (i.e. can be expressed as a
  --         projection functor from the partition and index), with
  --         simple indexing which can be analyzed by the optimizer.
  --      c. (Not yet implemented.) Neither, but (i) the region has no
  --         privileges, and (ii) it is provably a subregion of one of
  --         the other region arguments to the task (which must
  --         satisfy either (a) or (b)).
  --
  --  2. For region-typed arguments, compare for non-interference
  --     against previous arguments. Regions must be provably
  --     non-interfering with all previous arguments to pass this
  --     analysis.
  --
  --  3. For region-typed arguments, compare for non-interference
  --     against itself. This means that any arguments that are not
  --     provably loop-variant must be read-only (or reductions,
  --     atomic, simultaneous, etc.). Loop-variant arguments can
  --     additionally prove non-interference by demonstrating that
  --     they come from a disjoint partition, as long as indexing into
  --     said partition is provably disjoint.

  local args_provably = ast.IndexLaunchArgsProvably {
    invariant = terralib.newlist(),
    projectable = terralib.newlist(),
  }

  local is_constant_time = not node.annotations.constant_time_launch:is(ast.annotation.Forbid)

  local free_vars = terralib.newlist()
  local args_need_dynamic_check = terralib.newlist()

  -- Perform a simpler analysis if the expression is not a task launch
  if not call:is(ast.typed.expr.Call) then
    if call:is(ast.typed.expr.Fill) then
      if #preamble > 0 or #loop_vars > 0 then
        report_fail(call, "loop optimization failed: fill must not have any preamble statement")
      end

      if not analyze_is_side_effect_free(loop_cx, call.value) then
        report_fail(call.value, "loop optimization failed: fill value" ..
            " is not side-effect free")
        return
      end

      if not analyze_is_loop_invariant(loop_cx, call.value) then
        report_fail(call.value, "loop optimization failed: fill value" ..
            " is not provably invariant")
        return
      end

      local projection = call.dst.region
      if not analyze_is_projectable(loop_cx, projection) then
        report_fail(call, "loop optimization failed: fill target" ..
            " is not provably projectable")
        return
      end

      local partition_type = std.as_read(projection.value.expr_type)
      if not (partition_type and partition_type:is_disjoint() and
                analyze_index_noninterference_self(projection.index, loop_cx, loop_vars))
      then
        report_fail(call, "loop optimization failed: fill target" ..
            " interferes with itself")
        return
      end

      args_provably.invariant:insert(false)
      args_provably.projectable:insert(true)
    else
      -- TODO: Add index copies
      assert(false)
    end

  else
    local task = call.fn.value
    local param_types = task:get_type().parameters
    local args = call.args
    local regions_previously_used = terralib.newlist()
    local mapping = data.newmap()
    -- free variables referenced in loop variables (defined in loop preamble)
    -- must be defined for each arg since loop variables defined for each arg
    for _, var_stat in ipairs(loop_vars) do
       collect_free_variables(loop_cx, var_stat)
    end

    local free_vars_base = data.newmap()
    free_vars_base:insertall(loop_cx.free_variables)
    for i, arg in ipairs(args) do
      if not analyze_is_side_effect_free(loop_cx, arg) then
        report_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is not side-effect free")
        return
      end

      local arg_invariant = analyze_is_loop_invariant(loop_cx, arg)

      free_vars[i] = data.newmap()
      loop_cx.free_variables = data.newmap()
      loop_cx.free_variables:insertall(free_vars_base)
      collect_free_variables(loop_cx, arg)
      free_vars[i]:insertall(loop_cx.free_variables)

      local arg_projectable = false
      local partition_type

      local arg_type = std.as_read(arg.expr_type)
      -- XXX: This will break again if arg isn't unique for each argument,
      --      which can happen when de-duplicating AST nodes.
      assert(mapping[arg] == nil)
      mapping[arg] = param_types[i]
      -- Tests for conformance to index launch requirements.
      if std.is_region(arg_type) then
        if analyze_is_projectable(loop_cx, arg) then
          partition_type = std.as_read(
            util.get_base_indexed_node(strip_projection(arg)).expr_type):partition()
          assert(std.is_partition(partition_type))
          arg_projectable = true
        end

        if not (arg_projectable or arg_invariant) then
          report_fail(call, "loop optimization failed: argument " .. tostring(i) ..
              " is not provably projectable or invariant")
          return
        end
      end

      if std.is_phase_barrier(arg_type) then
        -- Phase barriers must be invariant, or must not be used as an arrival/wait.
        if not arg_invariant then
          for _, variables in task:get_conditions():values() do
            if variables[i] then
              report_fail(call, "loop optimization failed: argument " .. tostring(i) ..
                  " is not provably invariant")
              return
            end
          end
        end
      end

      if std.is_list(arg_type) and arg_type:is_list_of_regions() then
        -- FIXME: Deoptimize lists of regions for the moment. Lists
        -- would have to be (at a minimum) invariant though other
        -- restrictions may apply.
        report_fail(call, "loop optimization failed: argument " .. tostring(i) .. " is a list of regions")
        return
      end

      -- Tests for non-interference.
      if std.is_region(arg_type) and (not is_demand or not skip_interference_check) then
        do
          local passed, failures_i = analyze_noninterference_previous(
            loop_cx, task, arg, regions_previously_used, mapping)
          if not passed then
            if emit_dynamic_check then
              for _, failure in ipairs(failures_i) do
                if not (arg:is(ast.typed.expr.IndexAccess) and
                        args[failure]:is(ast.typed.expr.IndexAccess) and
                        get_base_partition_symbol(arg) == get_base_partition_symbol(args[failure]))
                then
                  report_fail(call, "loop optimization failed: argument " .. tostring(i) ..
                       " interferes with argument " .. tostring(failure))
                  return
                end
                args_need_dynamic_check:insert({i, failure })
              end
            else
              report_fail(call, "loop optimization failed: argument " .. tostring(i) ..
                   " interferes with argument " .. tostring(failures_i[1]))
              return
            end
          end
        end

        do
          local passed, needs_dynamic_check = analyze_noninterference_self(
            loop_cx, task, arg, partition_type, mapping, loop_vars)
          if not passed then
            if not (emit_dynamic_check and needs_dynamic_check) then
              report_fail(call, "loop optimization failed: argument " .. tostring(i) ..
                " interferes with itself")
              return
            else
              args_need_dynamic_check:insert(i)
            end
          end
        end
      end

      if not (arg_projectable or arg_invariant) then
        is_constant_time = false
        report_fail_ctl(call, "constant time launch failed: argument " .. tostring(i) ..
                      " is not provably projectable or invariant")
      end

      args_provably.invariant[i] = arg_invariant
      args_provably.projectable[i] = arg_projectable

      regions_previously_used[i] = nil
      if std.is_region(arg_type) then
        regions_previously_used[i] = arg
      end
    end
  end

  if #args_need_dynamic_check ~= 0 then
    report_pass("static loop optimization failed, emitting dynamic check")
  else
    report_pass("loop optimization succeeded")
  end

  return {
    preamble = preamble,
    call = call,
    reduce_lhs = reduce_lhs,
    reduce_op = reduce_op,
    args_provably = args_provably,
    is_constant_time = is_constant_time,
    free_variables = free_vars,
    loop_variables = loop_vars,
    args_need_dynamic_check = args_need_dynamic_check
  }
end

local function init_bitmask(bitmask, volume)
  local stats = terralib.newlist()

  local idx = std.newsymbol(int64, "i")
  local bitmask_i = util.mk_expr_index_access(util.mk_expr_id_rawref(bitmask), util.mk_expr_id(idx), std.rawref(&uint8))
  stats:insert(util.mk_stat_assignment(bitmask_i, util.mk_expr_constant(0, uint8)))

  local values = terralib.newlist()
  values:insert(util.mk_expr_constant(0, int64))
  values:insert(util.mk_expr_id(volume))

  return util.mk_stat_for_num(idx, values, util.mk_block(stats))
end

local function collapse_projection_functor(pf, dim, bounds)
  local index_types = { std.int1d, std.int2d, std.int3d, std.int4d,
    std.int5d, std.int6d, std.int7d, std.int8d, std.int9d }
  local dim_names = { "x", "y", "z", "w", "v", "u", "t", "s", "r" }

  local function mk_extent(dim)
    return util.mk_expr_binary("+",
      util.mk_expr_binary("-",
        util.mk_expr_field_access(util.mk_expr_field_access(bounds, "hi", index_types[dim]), dim_names[dim], int64),
        util.mk_expr_field_access(util.mk_expr_field_access(bounds, "lo", index_types[dim]), dim_names[dim], int64)),
      util.mk_expr_constant(1, int64))
  end

  local function recurse(dim, prev_extent, prev_ast)
    if dim == 0 then return prev_ast end

    local extent = util.mk_expr_binary("*", mk_extent(dim + 1), prev_extent)
    local ast = util.mk_expr_binary("+",
                 util.mk_expr_binary("*",
                   util.mk_expr_field_access(pf, dim_names[dim], int64),
                   extent),
                 prev_ast)
    return recurse(dim - 1, extent, ast)
  end

  if dim <= 1 then
    return pf
  else
    return recurse(
      dim - 1,
      util.mk_expr_constant(1, int64),
      util.mk_expr_field_access(pf, dim_names[dim], int64))
  end
end

local function collect_and_sort_args(args, call_args, task, param_region_types)
  -- Collect all args of the same partition or cross product together
  local collected = data.new_recursive_map(1)
  for _, arg in ipairs(args) do
    if type(arg) == "number" then
      local psym = get_base_partition_symbol(call_args[arg])
      collected[psym][arg] = true
    else
      -- Cross-check args are pairs
      local psym = get_base_partition_symbol(call_args[arg[1]])
      collected[psym][arg[1]] = true
      collected[psym][arg[2]] = true
    end
  end

  local array = data.newmap()
  for psym, args in collected:items() do
    array[psym] = terralib.newlist()
    for arg, v in args:items() do
      if v then array[psym]:insert(arg) end
    end
  end

  for _, args in array:items() do
    table.sort(args, function(p, q)
      local p_privileges, _, _, _, _ = std.find_task_privileges(param_region_types[p], task)
      local q_privileges, _, _, _, _ = std.find_task_privileges(param_region_types[q], task)

      local p_has_WR = p_privileges[1] == "reads_writes" or string.find(p_privileges[1], "reduces")
      local q_has_WR = q_privileges[1] == "reads_writes" or string.find(q_privileges[1], "reduces")

      if p_has_WR and not q_has_WR then
        return true
      elseif q_has_WR and not p_has_WR then
        return false
      end
      return p < q
    end)
  end

  return array
end

terra throw_dynamic_check_error(loc : rawstring, arg1 : uint8, arg2 : uint8)
  var stderr = std.c.fdopen(2, "w")
  if arg1 == arg2 then
    std.c.fprintf(stderr, [[Errors reported during runtime.
%s: loop optimization failed: argument %d interferes with itself
]], loc, arg1)
  else
    std.c.fprintf(stderr, [[Errors reported during runtime.
%s: loop optimization failed: argument %d interferes with argument %d
]], loc, arg1, arg2)
  end
  std.c.fflush(stderr)
  std.c.abort()
end

local function insert_dynamic_check(is_demand, args_need_dynamic_check, index_launch_ast, unopt_loop_ast)
  local check = terralib.newlist()

  local index_types = { [0] = std.int1d, std.int1d, std.int2d, std.int3d, std.int4d,
    std.int5d, std.int6d, std.int7d, std.int8d, std.int9d }
  local rect_types = { [0] = std.rect1d, std.rect1d, std.rect2d, std.rect3d, std.rect4d,
    std.rect5d, std.rect6d, std.rect7d, std.rect8d, std.rect9d }

  local call_args = index_launch_ast.call.args
  assert(#call_args < 256)

  local param_region_types = index_launch_ast.call.fn.value:get_type().parameters
  local task = index_launch_ast.call.fn.value

  local verdict = std.newsymbol(bool, "verdict")
  check:insert(util.mk_stat_var(verdict, bool, util.mk_expr_constant(false, bool)))

  local collected_args = collect_and_sort_args(
    args_need_dynamic_check, call_args, task, param_region_types)

  for _, args in collected_args:items() do
    local stats = terralib.newlist()

    local first_arg = util.get_base_indexed_node(strip_projection(call_args[args[1]]))
    local dim = get_partition_dim(first_arg)

    -- Generate the AST for var volume = p.colors.bounds:volume()
    local p_colors = util.mk_expr_field_access(
      util.mk_expr_id(first_arg.value), "colors", std.ispace(index_types[dim]))
    local p_bounds = util.mk_expr_field_access(p_colors, "bounds", rect_types[dim])
    local volume = std.newsymbol(int64, "volume")
    stats:insert(
     util.mk_stat_var(volume, int64, util.mk_expr_method_call(
       p_bounds, int64, "volume", terralib.newlist())))

    -- Malloc a bitmask of size volume and initialize it
    local bitmask_raw = std.newsymbol(&opaque, "bitmask_raw")
    local bitmask = std.newsymbol(&uint8, "bitmask")
    local malloc_call = util.mk_expr_call(
      std.c.malloc,
      util.mk_expr_cast(int64, util.mk_expr_id(volume)))
    stats:insert(
      util.mk_stat_var(bitmask_raw, &opaque, malloc_call))
    stats:insert(
      util.mk_stat_var(bitmask, &uint8, util.mk_expr_cast(&uint8, util.mk_expr_id(bitmask_raw))))
    stats:insert(init_bitmask(bitmask, volume))

    local value = std.newsymbol(int64, "value")
    stats:insert(util.mk_stat_var(value, int64))

    local conflict = std.newsymbol(uint8, "conflict")
    stats:insert(util.mk_stat_var(conflict, uint8, util.mk_expr_constant(0, uint8)))

    for _, arg in ipairs(args) do
      local duplicates_check = terralib.newlist()
      index_launch_ast.preamble:map(function(stat) duplicates_check:insert(stat) end)

      local _, index_expr = util.get_base_indexed_node(strip_projection(call_args[arg]))

      -- Assign: value = collapse(index_expr)
      duplicates_check:insert(
        util.mk_stat_assignment(
          util.mk_expr_id_rawref(value),
          collapse_projection_functor(index_expr, dim, p_bounds)))

      local then_block = terralib.newlist()

      local bitmask_value = util.mk_expr_index_access(util.mk_expr_id(bitmask),
        util.mk_expr_id(value), std.rawref(&uint8))
      then_block:insert(util.mk_stat_assignment(util.mk_expr_id_rawref(conflict), bitmask_value))

      local privileges, _, _, _, _ = std.find_task_privileges(param_region_types[arg], task)
      assert(#privileges == 1)
      if privileges[1] ~= "reads" then
        then_block:insert(util.mk_stat_assignment(bitmask_value, util.mk_expr_constant(arg, uint8)))
      end

      -- Issue an error here if we have to
      if is_demand then
        local abort = util.mk_stat_expr(util.mk_expr_call(throw_dynamic_check_error, terralib.newlist {
          util.mk_expr_constant(get_source_location(unopt_loop_ast), rawstring),
          util.mk_expr_constant(arg, uint8),
          util.mk_expr_id(conflict)
          }))
        then_block:insert(
          util.mk_stat_if(
            util.mk_expr_binary("~=", util.mk_expr_id(conflict), util.mk_expr_constant(0, uint8)),
            terralib.newlist { abort }))
      else
        local set_verdict = util.mk_stat_assignment(util.mk_expr_id_rawref(verdict),
          util.mk_expr_constant(true, bool))
        then_block:insert(
          util.mk_stat_if(
            util.mk_expr_binary("~=", util.mk_expr_id(conflict), util.mk_expr_constant(0, uint8)),
            terralib.newlist { set_verdict, util.mk_stat_break() }))
      end

      -- Bounds check
      local cond = util.mk_expr_binary("and",
        util.mk_expr_binary(">=", util.mk_expr_id(value), util.mk_expr_constant(0, int64)),
        util.mk_expr_binary("<", util.mk_expr_id(value), util.mk_expr_id(volume)))
      duplicates_check:insert(util.mk_stat_if(cond, then_block))

      -- Skip this check if a previous one failed
      if unopt_loop_ast.node_type:is(ast.typed.stat.ForNum) then
        local bounds = index_launch_ast.values
        stats:insert(
          util.mk_stat_if_else(
            util.mk_expr_id(verdict),
            terralib.newlist(),
            util.mk_stat_for_num(index_launch_ast.symbol, bounds, util.mk_block(duplicates_check))))
      else
        local bounds = index_launch_ast.value
        stats:insert(
          util.mk_stat_if_else(
            util.mk_expr_id(verdict),
            terralib.newlist(),
            util.mk_stat_for_list(index_launch_ast.symbol, bounds, util.mk_block(duplicates_check))))
      end
    end

    stats:insert(util.mk_stat_expr(util.mk_expr_call(std.c.free, util.mk_expr_id(bitmask))))

    if is_demand then
      check:insert(util.mk_stat_block(util.mk_block(stats)))
    else
      -- Skip this check if a previous one failed
      check:insert(util.mk_stat_if_else(util.mk_expr_id(verdict), terralib.newlist(), stats))
    end
  end

  -- In the demand case we know it's safe to optimize here, otherwise we need to check verdict
  if is_demand then
    check:insert(index_launch_ast)
  else
    check:insert(util.mk_stat_if_else(util.mk_expr_id(verdict), unopt_loop_ast, index_launch_ast))
  end

  return util.mk_stat_block(util.mk_block(check))
end

-- Height is the distance from the leaves up to the lowest parent
-- which is invariant within the loop.
local function find_invariant_prefix(cx, node, parent, height)
  local node = strip_projection(node)
  if not node:is(ast.typed.expr.IndexAccess) then
    return false
  end
  if not analyze_is_loop_invariant(cx, node) then
    return find_invariant_prefix(cx, node.value, node, height + 1)
  end
  return true, parent, height
end

local function get_node_at_height(arg, height)
  if height == 0 then
    return arg
  end
  return get_node_at_height(arg.value, height - 1)
end

local function replace_at_height(arg, height, replacement)
  if height == 0 then
    return replacement
  end
  return arg { value = replace_at_height(arg.value, height - 1, replacement) }
end

local function hoist_call_args(cx, hoisted, call)
  local collected = data.new_default_map(function() return terralib.newlist() end)
  for i, arg in ipairs(call.args) do
    if strip_projection(arg):is(ast.typed.expr.IndexAccess) and
       std.is_region(std.as_read(strip_projection(arg).expr_type))
    then
      collected[get_base_partition_symbol(arg)]:insert(i)
    end
  end

  local new_args = terralib.newlist()
  new_args:insertall(call.args)
  for _, indices in collected:values() do
    if not indices:find(
      function(i)
        return std.is_partition(std.as_read(
          util.get_base_indexed_node(strip_projection(new_args[i])).expr_type))
      end)
    then
      local heights = indices:map(
        function(i)
          local has_invariant_prefix, _, height = find_invariant_prefix(cx, new_args[i], true, 0)
          if not has_invariant_prefix then
            return -1
          end
          return height
        end)

      local continue = true
      for _, height in ipairs(heights) do
        if height == -1 then continue = false end
      end

      if continue then
        -- Every argument deriving from the same base cross product must have
        -- the same type. E.g. cp[i][i] and cp[i] are not allowed together.
        local licm_height = heights:reduce(math.max)

        if licm_height == 0 then
          -- Hoist the entire IndexAccess AST
          indices:app(function(i)
            local invariant = std.newsymbol(std.as_read(new_args[i].expr_type), "invariant")
            hoisted:insert(util.mk_stat_var(invariant, new_args[i].expr_type, new_args[i]))
            new_args[i] = util.mk_expr_id(invariant)
          end)
        else
          -- Hoist only a part of the IndexAccess AST
          indices:app(function(i)
            local parent = get_node_at_height(strip_projection(new_args[i]), licm_height - 1)
            local invariant = std.newsymbol(std.as_read(parent.value.expr_type), "invariant")
            hoisted:insert(util.mk_stat_var(invariant, parent.value.expr_type, parent.value))
            new_args[i] = replace_projection(
              new_args[i],
              replace_at_height(
                strip_projection(new_args[i]), licm_height, util.mk_expr_id(invariant)))
          end)
        end
      end
    end
  end
  return call { args = new_args }
end

-- This function does not detect all possible index launches,
-- just those we want to do LICM on.
local function maybe_index_launch(node)
  if #node.block.stats == 0 then
    return false
  end

  for i = 1, #node.block.stats - 1 do
    local stat = node.block.stats[i]
    if not stat:is(ast.typed.stat.Var) then
      return false
    end
  end

  local body = node.block.stats[#node.block.stats]
  if (body:is(ast.typed.stat.Reduce) and body.rhs:is(ast.typed.expr.Call)) or
     (body:is(ast.typed.stat.Expr) and body.expr:is(ast.typed.expr.Call))
  then
    return true
  end
  return false
end

local function licm(cx, node)
  local hoisted = terralib.newlist()

  if not maybe_index_launch(node) then
    return node, hoisted
  end

  local loop_cx = cx:new_local_scope()
  loop_cx:set_loop_index(node.symbol)
  loop_cx:add_loop_variable(node.symbol)

  local new_stats = terralib.newlist()
  for i = 1, #node.block.stats - 1 do
    local stat = node.block.stats[i]
    if analyze_is_loop_invariant(loop_cx, stat) then
      hoisted:insert(stat)
    else
      new_stats:insert(stat)
      loop_cx:add_loop_variable(stat.symbol)
    end
  end

  local body = node.block.stats[#node.block.stats]
  if body:is(ast.typed.stat.Expr) and body.expr:is(ast.typed.expr.Call) then
    new_stats:insert(body { expr = hoist_call_args(loop_cx, hoisted, body.expr) })
  elseif body:is(ast.typed.stat.Reduce) and body.rhs:is(ast.typed.expr.Call) then
    new_stats:insert(body { rhs = hoist_call_args(loop_cx, hoisted, body.rhs) })
  else
    assert(false)
  end

  node = node { block = node.block { stats = new_stats } }

  return node, hoisted
end

function optimize_index_launch.stat_for_num(cx, node)
  local report_pass = ignore
  local report_fail = report.info
  local is_demand = false
  if node.annotations.index_launch:is(ast.annotation.Demand) or
    node.annotations.constant_time_launch:is(ast.annotation.Demand)
  then
    report_pass = ignore
    report_fail = report.error
    is_demand = true
  end

  if node.annotations.index_launch:is(ast.annotation.Forbid) then
    if node.annotations.constant_time_launch:is(ast.annotation.Demand) then
      report_fail(node, "conflicting annotations: __constant_time_launch requires __index_launch")
    end
    return node
  end

  if node.values[3] and not (
    node.values[3]:is(ast.typed.expr.Constant) and
    node.values[3].value == 1)
  then
    report_fail(node, "loop optimization failed: stride not equal to 1")
    return node
  end

  local hoisted_stmts
  node, hoisted_stmts = licm(cx, node)
  local body = optimize_loop_body(cx, node, report_pass, report_fail)
  if not body then
    node = node {
      block = optimize_index_launches.block(cx, node.block),
    }
    if #hoisted_stmts == 0 then
      return node
    end
    -- Retain LICM even if we can't index launch
    hoisted_stmts:insert(node)
    return util.mk_stat_block(util.mk_block(hoisted_stmts))
  end

  -- If we reach here then either the self interference test failed,
  -- or the optimization was successful.
  local index_launch_ast = ast.typed.stat.IndexLaunchNum {
    symbol = node.symbol,
    values = node.values,
    preamble = body.preamble,
    call = body.call,
    reduce_lhs = body.reduce_lhs,
    reduce_op = body.reduce_op,
    reduce_task = false,
    args_provably = body.args_provably,
    is_constant_time = body.is_constant_time,
    free_vars = body.free_variables,
    loop_vars = body.loop_variables,
    annotations = node.annotations,
    span = node.span,
  }

  if #body.args_need_dynamic_check == 0 then
    if #hoisted_stmts == 0 then
      return index_launch_ast
    end
    hoisted_stmts:insert(index_launch_ast)
    return util.mk_stat_block(util.mk_block(hoisted_stmts))
  end

  local dynamic_check = insert_dynamic_check(
    is_demand, body.args_need_dynamic_check, index_launch_ast,
    node {
      block = optimize_index_launches.block(cx, node.block),
    })

  if #hoisted_stmts == 0 then
    return dynamic_check
  end
  hoisted_stmts:insert(dynamic_check)
  return util.mk_stat_block(util.mk_block(hoisted_stmts))
end

function optimize_index_launch.stat_for_list(cx, node)
  local report_pass = ignore
  local report_fail = report.info
  local is_demand = false
  if node.annotations.index_launch:is(ast.annotation.Demand) or
    node.annotations.constant_time_launch:is(ast.annotation.Demand)
  then
    report_pass = ignore
    report_fail = report.error
    is_demand = true
  end

  if node.annotations.index_launch:is(ast.annotation.Forbid) then
    if node.annotations.constant_time_launch:is(ast.annotation.Demand) then
      report_fail(node, "conflicting annotations: __constant_time_launch requires __index_launch")
    end
    return node
  end

  local value_type = std.as_read(node.value.expr_type)
  if not (std.is_rect_type(value_type) or
          std.is_ispace(value_type) or
          std.is_region(value_type))
  then
    report_fail(node, "loop optimization failed: domain is not a ispace or region")
    return node
  end

  local hoisted_stmts
  node, hoisted_stmts = licm(cx, node)
  local body = optimize_loop_body(cx, node, report_pass, report_fail)
  if not body then
    node = node {
      block = optimize_index_launches.block(cx, node.block),
    }
    if #hoisted_stmts == 0 then
      return node
    end
    -- Retain LICM even if we can't index launch
    hoisted_stmts:insert(node)
    return util.mk_stat_block(util.mk_block(hoisted_stmts))
  end

  -- If we reach here then either the self interference test failed,
  -- or the optimization was successful.
  local index_launch_ast = ast.typed.stat.IndexLaunchList {
    symbol = node.symbol,
    value = node.value,
    preamble = body.preamble,
    call = body.call,
    reduce_lhs = body.reduce_lhs,
    reduce_op = body.reduce_op,
    reduce_task = false,
    args_provably = body.args_provably,
    is_constant_time = body.is_constant_time,
    free_vars = body.free_variables,
    loop_vars = body.loop_variables,
    annotations = node.annotations,
    span = node.span,
  }

  if #body.args_need_dynamic_check == 0 then
    if #hoisted_stmts == 0 then
      return index_launch_ast
    end
    hoisted_stmts:insert(index_launch_ast)
    return util.mk_stat_block(util.mk_block(hoisted_stmts))
  end

  local dynamic_check = insert_dynamic_check(
    is_demand, body.args_need_dynamic_check, index_launch_ast, node {
      block = optimize_index_launches.block(cx, node.block),
    })

  if #hoisted_stmts == 0 then
    return dynamic_check
  end
  hoisted_stmts:insert(dynamic_check)
  return util.mk_stat_block(util.mk_block(hoisted_stmts))
end

local function do_nothing(cx, node) return node end

local optimize_index_launches_stat_table = {
  [ast.typed.stat.ForNum]    = optimize_index_launch.stat_for_num,
  [ast.typed.stat.ForList]   = optimize_index_launch.stat_for_list,
}

local optimize_index_launches_stat = ast.make_single_dispatch(
  optimize_index_launches_stat_table,
  {},
  do_nothing)

function optimize_index_launches.stat(cx, node)
  return optimize_index_launches_stat(cx)(node)
end

function optimize_index_launches.block(cx, node)
  return node {
    stats = node.stats:map(function(stat)
      return optimize_index_launches.stat(cx, stat)
    end)
  }
end

function optimize_index_launches.top_task(cx, node)
  if not node.body then return node end

  local cx = cx:new_task_scope(node.prototype:get_constraints())
  return ast.map_stat_postorder(optimize_index_launches_stat(cx), node)
end

function optimize_index_launches.top(cx, node)
  if node:is(ast.typed.top.Task) then
    -- Always run the optimization, just throw away the result if this
    -- is a leaf task. This is important because local tasks can be
    -- called inside of leaf tasks, but that cannot lead to an index
    -- launch.
    local result = optimize_index_launches.top_task(cx, node)
    if not node.config_options.leaf then
      return result
    end
  end

  return node
end

function optimize_index_launches.entry(node)
  local cx = context.new_global_scope({})
  return optimize_index_launches.top(cx, node)
end

optimize_index_launches.pass_name = "optimize_index_launches"

return optimize_index_launches
