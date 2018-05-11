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

-- Legion Parser

local parsing = require("parsing")
local ast = require("regent/ast")
local data = require("common/data")
local std = require("regent/std")

local parser = {}

function parser.lookaheadif(p, tok)
  return p:lookahead().type == tok
end

function parser.annotation_level(p)
  if p:nextif("__allow") then
    return ast.annotation.Allow { value = false }
  elseif p:nextif("__demand") then
    return ast.annotation.Demand { value = false }
  elseif p:nextif("__forbid") then
    return ast.annotation.Forbid { value = false }
  else
    return false
  end
end

function parser.annotation_values(p)
  if p:nextif("(") then
    p:expect("__unroll")
    p:expect("(")
    local value = p:expect(p.number)
    p:expect(")")
    if value.value ~= math.floor(value.value) then
      p:error("unroll factor should be an integer")
    end
    p:expect(")")
    return ast.annotation.Unroll { value = value }
  else
    return false
  end
end

function parser.annotation_name(p, required)
  if p:nextif("__cuda") then
    local values = p:annotation_values()
    return "cuda", values
  elseif p:nextif("__external") then
    return "external"
  elseif p:nextif("__inline") then
    return "inline"
  elseif p:nextif("__inner") then
    return "inner"
  elseif p:nextif("__leaf") then
    return "leaf"
  elseif p:nextif("__openmp") then
    return "openmp"
  elseif p:nextif("__optimize") then
    return "optimize"
  elseif p:nextif("__parallel") then
    return "parallel"
  elseif p:nextif("__spmd") then
    return "spmd"
  elseif p:nextif("__trace") then
    return "trace"
  elseif p:nextif("__vectorize") then
    return "vectorize"
  elseif required then
    p:error("expected annotation name")
  end
end

function parser.annotations(p, allow_expr, allow_stat)
  assert(allow_expr or allow_stat)
  local annotations = ast.default_annotations()

  local level = p:annotation_level()
  if not level then return annotations end

  local closed = false
  while level do
    p:expect("(")
    local name = p:annotation_name(true)
    annotations = annotations { [name] = level }

    while p:nextif(",") do
      local name = p:annotation_name(false)
      if name then
        annotations = annotations { [name] = level }
      elseif allow_expr then
        local expr = p:expr()
        p:expect(")")
        return expr { annotations = annotations }
      end
    end

    if allow_stat and p:nextif(")") then
      level = p:annotation_level()
      closed = not level
    end
  end

  if allow_stat and closed then
    return annotations
  else
    assert(allow_expr)
    -- Fail: This was supposed to be an expression, but we never saw
    -- the expression clause.
    p:expect(",")
  end
end

function parser.reduction_op(p, optional)
  if p:nextif("+") then
    return "+"
  elseif p:nextif("-") then
    return "-"
  elseif p:nextif("*") then
    return "*"
  elseif p:nextif("/") then
    return "/"
  elseif p:nextif("max") then
    return "max"
  elseif p:nextif("min") then
    return "min"
  elseif optional then
    return false
  else
    p:error("expected operator")
  end
end

function parser.field_names(p)
  local start = ast.save(p)
  local names_expr
  if p:nextif("[") then
    names_expr = p:luaexpr()
    p:expect("]")
  elseif p:nextif("ispace") then
    names_expr = "ispace"
  else
    names_expr = p:expect(p.name).value
  end
  return ast.unspecialized.FieldNames {
    names_expr = names_expr,
    span = ast.span(start, p),
  }
end

function parser.region_field(p)
  local start = ast.save(p)
  local field_name = p:field_names()
  local fields = false -- sentinel for all fields
  if p:nextif(".") then
    fields = p:region_fields()
  end
  return ast.unspecialized.region.Field {
    field_name = field_name,
    fields = fields,
    span = ast.span(start, p),
  }
end

function parser.region_fields(p)
  local fields = terralib.newlist()
  if p:nextif("{") then
    repeat
      if p:matches("}") then break end
      fields:insert(p:region_field())
    until not p:sep()
    p:expect("}")
  else
    fields:insert(p:region_field())
  end
  return fields
end

function parser.region_root(p)
  local start = ast.save(p)
  local region_name = p:expect(p.name).value
  p:ref(region_name)
  local fields = false -- sentinel for all fields
  if p:nextif(".") then
    fields = p:region_fields()
  end
  return ast.unspecialized.region.Root {
    region_name = region_name,
    fields = fields,
    span = ast.span(start, p),
  }
end

function parser.expr_region_root(p)
  local start = ast.save(p)
  local region = p:expr_prefix()
  local fields = false -- sentinel for all fields
  if p:nextif(".") then
    fields = p:region_fields()
  end
  return ast.unspecialized.expr.RegionRoot {
    region = region,
    fields = fields,
    annotations = ast.default_annotations(),
    span = ast.span(start, p),
  }
end

function parser.region_bare(p)
  local start = ast.save(p)
  local region_name = p:expect(p.name).value
  p:ref(region_name)
  return ast.unspecialized.region.Bare {
    region_name = region_name,
    span = ast.span(start, p),
  }
end

function parser.regions(p)
  local regions = terralib.newlist()
  repeat
    local region = p:region_root()
    regions:insert(region)
  until not p:nextif(",")
  return regions
end

function parser.condition_variable(p)
  local start = ast.save(p)
  local name = p:expect(p.name).value
  p:ref(name)
  return ast.unspecialized.ConditionVariable {
    name = name,
    span = ast.span(start, p),
  }
end

function parser.condition_variables(p)
  local variables = terralib.newlist()
  repeat
    variables:insert(p:condition_variable())
  until not p:nextif(",")
  return variables
end

function parser.is_privilege_kind(p)
  return p:matches("reads") or p:matches("writes") or p:matches("reduces")
end

function parser.privilege_kind(p)
  local start = ast.save(p)
  if p:nextif("reads") then
    return ast.privilege_kind.Reads {}
  elseif p:nextif("writes") then
    return ast.privilege_kind.Writes {}
  elseif p:nextif("reduces") then
    local op = p:reduction_op()
    return ast.privilege_kind.Reduces {
      op = op,
    }
  else
    p:error("expected privilege")
  end
end

function parser.is_coherence_kind(p)
  return p:matches("exclusive") or p:matches("atomic") or
    p:matches("simultaneous") or p:matches("relaxed")
end

function parser.coherence_kind(p)
  local start = ast.save(p)
  if p:nextif("exclusive") then
    return ast.coherence_kind.Exclusive {}
  elseif p:nextif("atomic") then
    return ast.coherence_kind.Atomic {}
  elseif p:nextif("simultaneous") then
    return ast.coherence_kind.Simultaneous {}
  elseif p:nextif("relaxed") then
    return ast.coherence_kind.Relaxed {}
  else
    p:error("expected coherence mode")
  end
end

function parser.is_flag_kind(p)
  return p:matches("no_access_flag")
end

function parser.flag_kind(p)
  local start = ast.save(p)
  if p:nextif("no_access_flag") then
    return ast.flag_kind.NoAccessFlag {}
  else
    p:error("expected flag")
  end
end

function parser.privilege_coherence_flag_kinds(p)
  local privileges = terralib.newlist()
  local coherence_modes = terralib.newlist()
  local flags = terralib.newlist()
  while not p:matches("(") do
    if p:is_privilege_kind() then
      privileges:insert(p:privilege_kind())
    elseif p:is_coherence_kind() then
      coherence_modes:insert(p:coherence_kind())
    elseif p:is_flag_kind() then
      flags:insert(p:flag_kind())
    else
      p:error("expected privilege or coherence mode")
    end
  end
  return privileges, coherence_modes, flags
end

function parser.privilege_coherence_flags(p)
  local start = ast.save(p)
  local privileges, coherence_modes, flags = p:privilege_coherence_flag_kinds()
  p:expect("(")
  local regions = p:regions()
  p:expect(")")

  local privilege = ast.unspecialized.Privilege {
    privileges = privileges,
    regions = regions,
    span = ast.span(start, p),
  }
  local coherence = ast.unspecialized.Coherence {
    coherence_modes = coherence_modes,
    regions = regions,
    span = ast.span(start, p),
  }
  local flag = ast.unspecialized.Flag {
    flags = flags,
    regions = regions,
    span = ast.span(start, p),
  }
  return privilege, coherence, flag
end

function parser.is_condition_kind(p)
  return p:matches("arrives") or p:matches("awaits")
end

function parser.condition_kind(p)
  local start = ast.save(p)
  if p:nextif("arrives") then
    return ast.condition_kind.Arrives {}
  elseif p:nextif("awaits") then
    return ast.condition_kind.Awaits {}
  else
    p:error("expected condition")
  end
end

function parser.condition_kinds(p)
  local conditions = terralib.newlist()
  while not p:matches("(") do
    conditions:insert(p:condition_kind())
  end
  return conditions
end

function parser.condition(p)
  local start = ast.save(p)
  local conditions = p:condition_kinds()
  p:expect("(")
  local variables = p:condition_variables()
  p:expect(")")

  return ast.unspecialized.Condition {
    conditions = conditions,
    variables = variables,
    span = ast.span(start, p),
  }
end

function parser.expr_condition(p)
  local start = ast.save(p)
  local conditions = p:condition_kinds()
  p:expect("(")
  local values = p:expr_list()
  p:expect(")")

  return ast.unspecialized.expr.Condition {
    conditions = conditions,
    values = values,
    annotations = ast.default_annotations(),
    span = ast.span(start, p),
  }
end

function parser.constraint_kind(p)
  local start = ast.save(p)
  if p:nextif("<=") then
    return ast.constraint_kind.Subregion {}
  elseif p:nextif("*") then
    return ast.constraint_kind.Disjointness {}
  else
    p:error("unexpected token in constraint")
  end
end

function parser.constraint(p)
  local start = ast.save(p)
  local lhs = p:region_bare()
  local op = p:constraint_kind()
  local rhs = p:region_bare()
  return ast.unspecialized.Constraint {
    lhs = lhs,
    op = op,
    rhs = rhs,
    span = ast.span(start, p),
  }
end

function parser.is_disjointness_kind(p)
  return p:matches("aliased") or p:matches("disjoint")
end

function parser.disjointness_kind(p)
  local start = ast.save(p)
  if p:nextif("aliased") then
    return ast.disjointness_kind.Aliased {}
  elseif p:nextif("disjoint") then
    return ast.disjointness_kind.Disjoint {}
  else
    p:error("expected disjointness")
  end
end

function parser.fence_kind(p)
  local start = ast.save(p)
  if p:nextif("__execution") then
    return ast.fence_kind.Execution {}
  elseif p:nextif("__mapping") then
    return ast.fence_kind.Mapping {}
  else
    p:error("expected fence kind (__execution or __mapping)")
  end
end

function parser.effect(p)
  local start = ast.save(p)
  p:expect("[")
  local effect_expr = p:luaexpr()
  p:expect("]")
  return ast.unspecialized.Effect {
    expr = effect_expr,
    span = ast.span(start, p),
  }
end

function parser.expr_prefix(p)
  local start = ast.save(p)
  if p:nextif("(") then
    local expr = p:expr()
    p:expect(")")
    return expr

  elseif p:nextif("[") then
    local expr = p:luaexpr()
    p:expect("]")
    return ast.unspecialized.expr.Escape {
      expr = expr,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:matches(p.name) then
    local name = p:expect(p.name).value
    p:ref(name)
    return ast.unspecialized.expr.ID {
      name = name,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("max") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local rhs = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Binary {
      op = "max",
      lhs = lhs,
      rhs = rhs,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("min") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local rhs = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Binary {
      op = "min",
      lhs = lhs,
      rhs = rhs,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__context") then
    p:expect("(")
    p:expect(")")
    return ast.unspecialized.expr.RawContext {
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__fields") then
    p:expect("(")
    local region = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.RawFields {
      region = region,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__physical") then
    p:expect("(")
    local region = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.RawPhysical {
      region = region,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__delete") then
    p:expect("(")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.RawDelete {
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__runtime") then
    p:expect("(")
    p:expect(")")
    return ast.unspecialized.expr.RawRuntime {
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("__raw") then
    p:expect("(")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.RawValue {
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("isnull") then
    p:expect("(")
    local pointer = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Isnull {
      pointer = pointer,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("new") then
    p:expect("(")
    local pointer_type_expr = p:luaexpr()
    local extent = false
    if p:nextif(",") then
      extent = p:expr()
    end
    p:expect(")")
    return ast.unspecialized.expr.New {
      pointer_type_expr = pointer_type_expr,
      extent = extent,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("null") then
    p:expect("(")
    local pointer_type_expr = p:luaexpr()
    p:expect(")")
    return ast.unspecialized.expr.Null {
      pointer_type_expr = pointer_type_expr,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("dynamic_cast") then
    p:expect("(")
    local type_expr = p:luaexpr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.DynamicCast {
      type_expr = type_expr,
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("static_cast") then
    p:expect("(")
    local type_expr = p:luaexpr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.StaticCast {
      type_expr = type_expr,
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("unsafe_cast") then
    p:expect("(")
    local type_expr = p:luaexpr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.UnsafeCast {
      type_expr = type_expr,
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("ispace") then
    p:expect("(")
    local index_type_expr = p:luaexpr()
    p:expect(",")
    local extent = p:expr()
    local start_at = false
    if p:nextif(",") then
      start_at = p:expr()
    end
    p:expect(")")
    return ast.unspecialized.expr.Ispace {
      index_type_expr = index_type_expr,
      extent = extent,
      start = start_at,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("region") then
    p:expect("(")
    local ispace = p:expr()
    p:expect(",")
    local fspace_type_expr = p:luaexpr()
    p:expect(")")
    return ast.unspecialized.expr.Region {
      ispace = ispace,
      fspace_type_expr = fspace_type_expr,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("partition") then
    p:expect("(")
    if p:is_disjointness_kind() then
      local disjointness = p:disjointness_kind()
      p:expect(",")
      local region = p:expr()
      p:expect(",")
      local coloring = p:expr()
      local colors = false
      if not p:matches(")") then
        p:expect(",")
        colors = p:expr()
      end
      p:expect(")")
      return ast.unspecialized.expr.Partition {
        disjointness = disjointness,
        region = region,
        coloring = coloring,
        colors = colors,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }
    elseif p:nextif("equal") then
      p:expect(",")
      local region = p:expr()
      p:expect(",")
      local colors = p:expr()
      p:expect(")")
      return ast.unspecialized.expr.PartitionEqual {
        region = region,
        colors = colors,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }
    else
      local region = p:expr_region_root()
      p:expect(",")
      local colors = p:expr()
      p:expect(")")
      return ast.unspecialized.expr.PartitionByField {
        region = region,
        colors = colors,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }
    end

  elseif p:nextif("image") then
    p:expect("(")
    local parent = p:expr()
    p:expect(",")
    local partition = p:expr()
    p:expect(",")
    local region = p:expr_region_root()
    p:expect(")")
    return ast.unspecialized.expr.Image {
      parent = parent,
      partition = partition,
      region = region,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("preimage") then
    p:expect("(")
    local parent = p:expr()
    p:expect(",")
    local partition = p:expr()
    p:expect(",")
    local region = p:expr_region_root()
    p:expect(")")
    return ast.unspecialized.expr.Preimage {
      parent = parent,
      partition = partition,
      region = region,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("cross_product") then
    p:expect("(")
    local args = p:expr_list()
    p:expect(")")
    return ast.unspecialized.expr.CrossProduct {
      args = args,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("cross_product_array") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local disjointness = p:disjointness_kind()
    p:expect(",")
    local colorings = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.CrossProductArray {
      lhs = lhs,
      disjointness = disjointness,
      colorings = colorings,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_slice_partition") then
    p:expect("(")
    local partition = p:expr()
    p:expect(",")
    local indices = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.ListSlicePartition {
      partition = partition,
      indices = indices,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_duplicate_partition") then
    p:expect("(")
    local partition = p:expr()
    p:expect(",")
    local indices = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.ListDuplicatePartition {
      partition = partition,
      indices = indices,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_cross_product") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local rhs = p:expr()
    local shallow = false -- Default is false.
    if p:nextif(",") then
      if p:nextif("true") then
        shallow = true
      elseif p:nextif("false") then
        shallow = false
      else
        p:error("expected true or false")
      end
    end
    p:expect(")")
    return ast.unspecialized.expr.ListCrossProduct {
      lhs = lhs,
      rhs = rhs,
      shallow = shallow,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_cross_product_complete") then
    p:expect("(")
    local lhs = p:expr()
    p:expect(",")
    local product = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.ListCrossProductComplete {
      lhs = lhs,
      product = product,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_phase_barriers") then
    p:expect("(")
    local product = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.ListPhaseBarriers {
      product = product,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_invert") then
    p:expect("(")
    local rhs = p:expr()
    p:expect(",")
    local product = p:expr()
    p:expect(",")
    local barriers = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.ListInvert {
      rhs = rhs,
      product = product,
      barriers = barriers,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_range") then
    p:expect("(")
    local range_start = p:expr()
    p:expect(",")
    local range_stop = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.ListRange {
      start = range_start,
      stop = range_stop,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_ispace") then
    p:expect("(")
    local ispace = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.ListIspace {
      ispace = ispace,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("list_from_element") then
    p:expect("(")
    local list = p:expr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.ListFromElement {
      list = list,
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("phase_barrier") then
    p:expect("(")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.PhaseBarrier {
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("dynamic_collective") then
    p:expect("(")
    local value_type_expr = p:luaexpr()
    p:expect(",")
    local op = p:reduction_op()
    p:expect(",")
    local arrivals = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.DynamicCollective {
      value_type_expr = value_type_expr,
      arrivals = arrivals,
      op = op,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("dynamic_collective_get_result") then
    p:expect("(")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.DynamicCollectiveGetResult {
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("advance") then
    p:expect("(")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Advance {
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("adjust") then
    p:expect("(")
    local barrier = p:expr()
    p:expect(",")
    local value = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Adjust {
      barrier = barrier,
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("arrive") then
    p:expect("(")
    local barrier = p:expr()
    local value = false
    if p:nextif(",") then
      value = p:expr()
    end
    p:expect(")")
    return ast.unspecialized.expr.Arrive {
      barrier = barrier,
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("await") then
    p:expect("(")
    local barrier = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.Await {
      barrier = barrier,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("copy") then
    p:expect("(")
    local src = p:expr_region_root()
    p:expect(",")
    local dst = p:expr_region_root()
    local op = false
    local try = p:nextif(",")
    if try then
      op = p:reduction_op(true)
    end
    local conditions = terralib.newlist()
    if (try and not op) or p:nextif(",") then
      repeat
        conditions:insert(p:expr_condition())
      until not p:nextif(",")
    end
    p:expect(")")
    return ast.unspecialized.expr.Copy {
      src = src,
      dst = dst,
      op = op,
      conditions = conditions,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("fill") then
    p:expect("(")
    local dst = p:expr_region_root()
    p:expect(",")
    local value = p:expr()
    local conditions = terralib.newlist()
    if p:nextif(",") then
      repeat
        conditions:insert(p:expr_condition())
      until not p:nextif(",")
    end
    p:expect(")")
    return ast.unspecialized.expr.Fill {
      dst = dst,
      value = value,
      conditions = conditions,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("acquire") then
    p:expect("(")
    local region = p:expr_region_root()
    local conditions = terralib.newlist()
    if p:nextif(",") then
      repeat
        conditions:insert(p:expr_condition())
      until not p:nextif(",")
    end
    p:expect(")")
    return ast.unspecialized.expr.Acquire {
      region = region,
      conditions = conditions,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("release") then
    p:expect("(")
    local region = p:expr_region_root()
    local conditions = terralib.newlist()
    if p:nextif(",") then
      repeat
        conditions:insert(p:expr_condition())
      until not p:nextif(",")
    end
    p:expect(")")
    return ast.unspecialized.expr.Release {
      region = region,
      conditions = conditions,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("attach") then
    p:expect("(")
    p:expect("hdf5")
    p:expect(",")
    local region = p:expr_region_root()
    p:expect(",")
    local filename = p:expr()
    p:expect(",")
    local mode = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.AttachHDF5 {
      region = region,
      filename = filename,
      mode = mode,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("detach") then
    p:expect("(")
    p:expect("hdf5")
    p:expect(",")
    local region = p:expr_region_root()
    p:expect(")")
    return ast.unspecialized.expr.DetachHDF5 {
      region = region,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("allocate_scratch_fields") then
    p:expect("(")
    local region = p:expr_region_root()
    p:expect(")")
    return ast.unspecialized.expr.AllocateScratchFields {
      region = region,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("with_scratch_fields") then
    p:expect("(")
    local region = p:expr_region_root()
    p:expect(",")
    local field_ids = p:expr()
    p:expect(")")
    return ast.unspecialized.expr.WithScratchFields {
      region = region,
      field_ids = field_ids,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  else
    p:error("unexpected token in expression")
  end
end

function parser.field(p)
  local start = ast.save(p)
  if p:matches(p.name) and p:lookaheadif("=") then
    local name = p:expect(p.name).value
    p:expect("=")
    local value = p:expr()
    return ast.unspecialized.expr.CtorRecField {
      name_expr = function(env) return name end,
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("[") then
    local name_expr = p:luaexpr()
    p:expect("]")
    if p:nextif("=") then
      local value = p:expr()
      return ast.unspecialized.expr.CtorRecField {
        name_expr = name_expr,
        value = value,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }
    else
      local value = ast.unspecialized.expr.Escape {
        expr = name_expr,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }
      return ast.unspecialized.expr.CtorListField {
        value = value,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }
    end

  else
    local value = p:expr()
    return ast.unspecialized.expr.CtorListField {
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }
  end
end

function parser.sep(p)
  return p:nextif(",") or p:nextif(";")
end

function parser.expr_ctor(p)
  local start = ast.save(p)
  local fields = terralib.newlist()
  p:expect("{")
  repeat
    if p:matches("}") then break end
    local field = p:field()
    fields:insert(field)
  until not p:sep()
  p:expect("}")

  return ast.unspecialized.expr.Ctor {
    fields = fields,
    annotations = ast.default_annotations(),
    span = ast.span(start, p),
  }
end

function parser.fnargs(p)
  if p:nextif("(") then
    local args = terralib.newlist()
    local conditions = terralib.newlist()
    if not p:matches(")") then
      repeat
        if p:is_condition_kind() then
          break
        end
        args:insert(p:expr())
      until not p:nextif(",")

      if not p:matches(")") then
        repeat
          conditions:insert(p:expr_condition())
        until not p:nextif(",")
      end
    end
    p:expect(")")
    return args, conditions

  elseif p:matches("{") then
    local arg = p:expr_ctor()
    return terralib.newlist({arg}), terralib.newlist()

  elseif p:matches(p.string) then
    local arg = p:expr_simple()
    return terralib.newlist({arg}), terralib.newlist()

  else
    p:error("unexpected token in fnargs expression")
  end
end

function parser.expr_primary_continuation(p, expr)
  local start = ast.save(p)

  while true do
    if p:nextif(".") then
      local field_names = terralib.newlist()
      if p:nextif("{") then
        repeat
          if p:matches("}") then break end
          field_names:insert(p:field_names())
        until not p:sep()
        p:expect("}")
      else
        field_names:insert(p:field_names())
      end
      expr = ast.unspecialized.expr.FieldAccess {
        value = expr,
        field_names = field_names,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }

    elseif p:nextif("[") then
      local index = p:expr()
      p:expect("]")
      expr = ast.unspecialized.expr.IndexAccess {
        value = expr,
        index = index,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }

    elseif p:nextif(":") then
      local method_name = p:expect(p.name).value
      local args, conditions = p:fnargs()
      if #conditions > 0 then
        p:error("method call cannot have conditions")
      end
      expr = ast.unspecialized.expr.MethodCall {
        value = expr,
        method_name = method_name,
        args = args,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }

    elseif p:matches("(") or p:matches("{") or p:matches(p.string) then
      local args, conditions = p:fnargs()
      expr = ast.unspecialized.expr.Call {
        fn = expr,
        args = args,
        conditions = conditions,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }

    else
      break
    end
  end

  return expr
end

function parser.expr_primary(p)
  local expr = p:expr_prefix()
  return p:expr_primary_continuation(expr)
end

function parser.expr_simple(p)
  local annotations = p:annotations(true, false)
  if annotations:is(ast.unspecialized.expr) then
    return annotations
  end

  local start = ast.save(p)
  if p:matches(p.number) then
    local token = p:expect(p.number)
    return ast.unspecialized.expr.Constant {
      value = token.value,
      expr_type = token.valuetype,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:matches(p.string) then
    local token = p:expect(p.string)
    return ast.unspecialized.expr.Constant {
      value = token.value,
      expr_type = rawstring,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("true") then
    return ast.unspecialized.expr.Constant {
      value = true,
      expr_type = bool,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:nextif("false") then
    return ast.unspecialized.expr.Constant {
      value = false,
      expr_type = bool,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }

  elseif p:matches("{") then
    return p:expr_ctor()

  else
    return p:expr_primary()
  end
end

parser.expr_unary = function(precedence)
  return function(p)
    local start = ast.save(p)
    local op = p:next().type
    local rhs = p:expr(precedence)
    if op == "@" then
      return ast.unspecialized.expr.Deref {
        value = rhs,
        annotations = ast.default_annotations(),
        span = ast.span(start, p),
      }
    end
    return ast.unspecialized.expr.Unary {
      op = op,
      rhs = rhs,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }
  end
end

parser.expr_binary_left = function(p, lhs)
  local start = lhs.span.start
  local op = p:next().type
  local rhs = p:expr(op)
  return ast.unspecialized.expr.Binary {
    op = op,
    lhs = lhs,
    rhs = rhs,
    annotations = ast.default_annotations(),
    span = ast.span(start, p),
  }
end

parser.expr = parsing.Pratt()
  :prefix("@", parser.expr_unary(50))
  :prefix("-", parser.expr_unary(50))
  :prefix("not", parser.expr_unary(50))
  :infix("*", 40, parser.expr_binary_left)
  :infix("/", 40, parser.expr_binary_left)
  :infix("%", 40, parser.expr_binary_left)
  :infix("+", 30, parser.expr_binary_left)
  :infix("-", 30, parser.expr_binary_left)
  :infix("<", 20, parser.expr_binary_left)
  :infix(">", 20, parser.expr_binary_left)
  :infix("<=", 20, parser.expr_binary_left)
  :infix(">=", 20, parser.expr_binary_left)
  :infix("==", 20, parser.expr_binary_left)
  :infix("~=", 20, parser.expr_binary_left)
  :infix("&", 15, parser.expr_binary_left)
  :infix("|", 12, parser.expr_binary_left)
  :infix("and", 10, parser.expr_binary_left)
  :infix("or", 10, parser.expr_binary_left)
  :prefix(parsing.default, parser.expr_simple)

function parser.expr_lhs(p)
  local start = ast.save(p)
  if p:nextif("@") then
    local value = p:expr(50) -- Precedence for unary @
    return ast.unspecialized.expr.Deref {
      value = value,
      annotations = ast.default_annotations(),
      span = ast.span(start, p),
    }
  else
    return p:expr_primary()
  end
end

function parser.expr_list(p)
  local exprs = terralib.newlist()
  repeat
    exprs:insert(p:expr())
  until not p:nextif(",")
  return exprs
end

function parser.block(p)
  local start = ast.save(p)
  local block = terralib.newlist()
  while not (p:matches("end") or p:matches("elseif") or
             p:matches("else") or p:matches("until")) do
    local stat = p:stat()
    block:insert(stat)
    p:nextif(";")
  end
  return ast.unspecialized.Block {
    stats = block,
    span = ast.span(start, p),
  }
end

function parser.stat_if(p, annotations)
  local start = ast.save(p)
  p:expect("if")
  local cond = p:expr()
  p:expect("then")
  local then_block = p:block()
  local elseif_blocks = terralib.newlist()
  local elseif_start = ast.save(p)
  while p:nextif("elseif") do
    local elseif_cond = p:expr()
    p:expect("then")
    local elseif_block = p:block()
    elseif_blocks:insert(ast.unspecialized.stat.Elseif {
        cond = elseif_cond,
        block = elseif_block,
        annotations = annotations,
        span = ast.span(elseif_start, p),
    })
    elseif_start = ast.save(p)
  end
  local else_block = ast.unspecialized.Block {
    stats = terralib.newlist(),
    span = ast.empty_span(p),
  }
  if p:nextif("else") then
    else_block = p:block()
  end
  p:expect("end")
  return ast.unspecialized.stat.If {
    cond = cond,
    then_block = then_block,
    elseif_blocks = elseif_blocks,
    else_block = else_block,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_while(p, annotations)
  local start = ast.save(p)
  p:expect("while")
  local cond = p:expr()
  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.While {
    cond = cond,
    block = block,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_for_num(p, start, name, type_expr, annotations)
  local values = p:expr_list()

  if #values < 2 or #values > 3 then
    p:error("for loop over numbers requires two or three values")
  end

  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.ForNum {
    name = name,
    type_expr = type_expr,
    values = values,
    block = block,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_for_list(p, start, name, type_expr, annotations)
  local value = p:expr()

  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.ForList {
    name = name,
    type_expr = type_expr,
    value = value,
    block = block,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_for(p, annotations)
  local start = ast.save(p)
  p:expect("for")

  local name, type_expr
  if p:nextif("[") then
    name = p:luaexpr()
    type_expr = false
    p:expect("]")
  else
    name = p:expect(p.name).value
    if p:nextif(":") then
      type_expr = p:luaexpr()
    else
      type_expr = false
    end
  end

  if p:nextif("=") then
    return p:stat_for_num(start, name, type_expr, annotations)
  elseif p:nextif("in") then
    return p:stat_for_list(start, name, type_expr, annotations)
  else
    p:error("expected = or in")
  end
end

function parser.stat_repeat(p, annotations)
  local start = ast.save(p)
  p:expect("repeat")
  local block = p:block()
  p:expect("until")
  local until_cond = p:expr()
  return ast.unspecialized.stat.Repeat {
    block = block,
    until_cond = until_cond,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_must_epoch(p, annotations)
  local start = ast.save(p)
  p:expect("must_epoch")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.MustEpoch {
    block = block,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_block(p, annotations)
  local start = ast.save(p)
  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.Block {
    block = block,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_var_unpack(p, start, annotations)
  p:expect("{")
  local names = terralib.newlist()
  local fields = terralib.newlist()
  repeat
    local name = p:expect(p.name).value
    names:insert(name)
    if p:nextif("=") then
      fields:insert(p:expect(p.name).value)
    else
      fields:insert(name)
    end
  until not p:nextif(",")
  p:expect("}")
  p:expect("=")
  local value = p:expr()
  return ast.unspecialized.stat.VarUnpack {
    var_names = names,
    fields = fields,
    value = value,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_var(p, annotations)
  local start = ast.save(p)
  p:expect("var")
  if p:matches("{") then
    return p:stat_var_unpack(start, annotations)
  end

  local names = terralib.newlist()
  local type_exprs = terralib.newlist()
  repeat
    if p:nextif("[") then
      names:insert(p:luaexpr())
      type_exprs:insert(false)
      p:expect("]")
    else
      names:insert(p:expect(p.name).value)
      if p:nextif(":") then
        type_exprs:insert(p:luaexpr())
      else
        type_exprs:insert(false)
      end
    end
  until not p:nextif(",")
  local values = terralib.newlist()
  if p:nextif("=") then
    values = p:expr_list()
  end
  return ast.unspecialized.stat.Var {
    var_names = names,
    type_exprs = type_exprs,
    values = values,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_return(p, annotations)
  local start = ast.save(p)
  p:expect("return")
  local value = false
  if not (p:matches("end") or p:matches("elseif") or
          p:matches("else") or p:matches("until"))
  then
    value = p:expr()
  end
  return ast.unspecialized.stat.Return {
    value = value,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_break(p, annotations)
  local start = ast.save(p)
  p:expect("break")
  return ast.unspecialized.stat.Break {
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_raw_delete(p, annotations)
  local start = ast.save(p)
  p:expect("__delete")
  p:expect("(")
  local value = p:expr()
  p:expect(")")
  return ast.unspecialized.stat.RawDelete {
    value = value,
    annotations = ast.default_annotations(),
    span = ast.span(start, p),
  }
end

function parser.stat_fence(p, annotations)
  local start = ast.save(p)
  p:expect("__fence")
  p:expect("(")
  local kind = p:fence_kind()
  local blocking = false
  if p:nextif(",") then
    p:expect("__block")
    blocking = true
  end
  p:expect(")")
  return ast.unspecialized.stat.Fence {
    kind = kind,
    blocking = blocking,
    annotations = ast.default_annotations(),
    span = ast.span(start, p),
  }
end

function parser.stat_parallelize_with(p, annotations)
  local start = ast.save(p)
  p:expect("__parallelize_with")
  local hints = terralib.newlist()
  hints:insert(p:expr())
  while p:nextif(",") do
    hints:insert(p:expr())
  end
  p:expect("do")
  local block = p:block()
  p:expect("end")
  return ast.unspecialized.stat.ParallelizeWith {
    hints = hints,
    block = block,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.stat_expr_assignment(p, start, first_lhs, annotations)
  local lhs = terralib.newlist()
  lhs:insert(first_lhs)
  while p:nextif(",") do
    lhs:insert(p:expr_lhs())
  end

  local op
  -- Hack: Terra's lexer doesn't understand += as a single operator so
  -- for the moment read it as + followed by =.
  if p:lookaheadif("=") then
    op = p:reduction_op(true)
    p:expect("=")
  else
    p:expect("=")
  end

  local rhs = p:expr_list()
  if op then
    return ast.unspecialized.stat.Reduce {
      op = op,
      lhs = lhs,
      rhs = rhs,
      annotations = annotations,
      span = ast.span(start, p),
    }
  else
    return ast.unspecialized.stat.Assignment {
      lhs = lhs,
      rhs = rhs,
      annotations = annotations,
      span = ast.span(start, p),
    }
  end
end

function parser.stat_expr_escape(p, annotations)
  local start = ast.save(p)

  p:expect("[")
  local value = p:luaexpr()
  p:expect("]")

  return ast.unspecialized.expr.Escape {
    expr = value,
    annotations = ast.default_annotations(),
    span = ast.span(start, p),
  }
end

function parser.stat_expr(p, annotations)
  local start = ast.save(p)

  local quoted_maybe_stat = false
  local first_lhs
  if p:matches("[") then
    first_lhs = p:stat_expr_escape(start, annotations)
    if p:matches(".") or p:matches("[") or p:matches(":") or
      p:matches("(") or p:matches("{") or p:matches(p.string)
    then
      first_lhs = p:expr_primary_continuation(first_lhs)
    else
      quoted_maybe_stat = true
    end
  else
    first_lhs = p:expr_lhs()
  end

  if p:matches(",") or
    p:matches("=") or
    (p:matches("+") and p:lookaheadif("=")) or
    (p:matches("-") and p:lookaheadif("=")) or
    (p:matches("*") and p:lookaheadif("=")) or
    (p:matches("/") and p:lookaheadif("=")) or
    (p:matches("max") and p:lookaheadif("=")) or
    (p:matches("min") and p:lookaheadif("="))
  then
    return p:stat_expr_assignment(start, first_lhs, annotations)
  elseif quoted_maybe_stat then
    return ast.unspecialized.stat.Escape {
      expr = first_lhs.expr,
      annotations = annotations,
      span = ast.span(start, p),
    }
  else
    return ast.unspecialized.stat.Expr {
      expr = first_lhs,
      annotations = annotations,
      span = ast.span(start, p),
    }
  end
end

function parser.stat(p)
  local annotations = p:annotations(true, true)
  if annotations:is(ast.unspecialized.expr) then
    return ast.unspecialized.stat.Expr {
      expr = annotations,
      annotations = ast.default_annotations(),
      span = annotations.span,
    }
  end

  if p:matches("if") then
    return p:stat_if(annotations)

  elseif p:matches("while") then
    return p:stat_while(annotations)

  elseif p:matches("for") then
    return p:stat_for(annotations)

  elseif p:matches("repeat") then
    return p:stat_repeat(annotations)

  elseif p:matches("must_epoch") then
    return p:stat_must_epoch(annotations)

  elseif p:matches("do") then
    return p:stat_block(annotations)

  elseif p:matches("var") then
    return p:stat_var(annotations)

  elseif p:matches("return") then
    return p:stat_return(annotations)

  elseif p:matches("break") then
    return p:stat_break(annotations)

  elseif p:matches("__delete") then
    return p:stat_raw_delete(annotations)

  elseif p:matches("__fence") then
    return p:stat_fence(annotations)

  elseif p:matches("__parallelize_with") then
    return p:stat_parallelize_with(annotations)

  else
    return p:stat_expr(annotations)
  end
end

function parser.top_task_name(p)
  local name = terralib.newlist()
  repeat
    name:insert(p:expect(p.name).value)
  until not p:nextif(".")
  return data.newtuple(unpack(name))
end

function parser.top_task_params(p)
  p:expect("(")
  local params = terralib.newlist()
  if not p:matches(")") then
    repeat
      local start = ast.save(p)
      local param_name, param_type
      if p:nextif("[") then
        param_name = p:luaexpr()
        param_type = false
        p:expect("]")
      else
        param_name = p:expect(p.name).value
        p:expect(":")
        param_type = p:luaexpr()
      end
      params:insert(ast.unspecialized.top.TaskParam {
          param_name = param_name,
          type_expr = param_type,
          annotations = ast.default_annotations(),
          span = ast.span(start, p),
      })
    until not p:nextif(",")
  end
  p:expect(")")
  return params
end

function parser.top_task_return(p)
  if p:nextif(":") then
    return p:luaexpr()
  end
  return function(env) return std.untyped end
end

function parser.top_task_effects(p, declaration)
  local effect_exprs = terralib.newlist()
  if p:nextif("where") then
    repeat
      if p:matches("[") then
        effect_exprs:insert(p:effect())
      elseif p:is_privilege_kind() or p:is_coherence_kind() or p:is_flag_kind() then
        local privilege, coherence, flag = p:privilege_coherence_flags()
        effect_exprs:insert(privilege)
        effect_exprs:insert(coherence)
        effect_exprs:insert(flag)
      elseif p:is_condition_kind() then
        effect_exprs:insert(p:condition())
      else
        effect_exprs:insert(p:constraint())
      end
    until not p:nextif(",")
    if not declaration then
      p:expect("do")
    else
      p:expect("end")
    end
  end
  return effect_exprs
end

function parser.top_task(p, annotations)
  local start = ast.save(p)
  local declaration = false
  if p:nextif("extern") then
    declaration = true
  end
  p:expect("task")
  local name = p:top_task_name()
  local params = p:top_task_params()
  local return_type = p:top_task_return()
  local effects = p:top_task_effects(declaration)
  local body = false
  if not declaration then
    body = p:block()
    p:expect("end")
  end

  return ast.unspecialized.top.Task {
    name = name,
    params = params,
    return_type_expr = return_type,
    effect_exprs = effects,
    body = body,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.top_fspace_params(p)
  local params = terralib.newlist()
  if p:nextif("(") then
    if not p:matches(")") then
      repeat
        local start = ast.save(p)
        local param_name = p:expect(p.name).value
        p:expect(":")
        local param_type = p:luaexpr()
        params:insert(ast.unspecialized.top.FspaceParam {
          param_name = param_name,
          type_expr = param_type,
          annotations = ast.default_annotations(),
          span = ast.span(start, p),
        })
      until not p:nextif(",")
    end
    p:expect(")")
  end
  return params
end

function parser.top_fspace_fields(p)
  local fields = terralib.newlist()
  p:expect("{")
  repeat
    if p:matches("}") then break end

    local start = ast.save(p)
    local field_names = terralib.newlist()
    if p:nextif("{") then
      repeat
        if p:matches("}") then break end
        field_names:insert(p:expect(p.name).value)
      until not p:nextif(",")
      p:expect("}")
    else
      field_names:insert(p:expect(p.name).value)
    end
    p:expect(":")
    local field_type = p:luaexpr()

    fields:insertall(
      field_names:map(
        function(field_name)
          return ast.unspecialized.top.FspaceField {
            field_name = field_name,
            type_expr = field_type,
            annotations = ast.default_annotations(),
            span = ast.span(start, p),
          }
        end))
  until not p:sep()
  p:expect("}")
  return fields
end

function parser.top_fspace_constraints(p)
  local constraints = terralib.newlist()
  if p:nextif("where") then
    repeat
      constraints:insert(p:constraint())
    until not p:nextif(",")
    p:expect("end")
  end
  return constraints
end

function parser.top_fspace(p, annotations)
  local start = ast.save(p)
  p:expect("fspace")
  local name = p:expect(p.name).value
  local params = p:top_fspace_params()
  local fields = p:top_fspace_fields()
  local constraints = p:top_fspace_constraints()

  return ast.unspecialized.top.Fspace {
    name = name,
    params = params,
    fields = fields,
    constraints = constraints,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.top_stat(p)
  local annotations = p:annotations(false, true)

  if p:matches("extern") or p:matches("task") then
    return p:top_task(annotations)

  elseif p:matches("fspace") then
    return p:top_fspace(annotations)

  else
    p:error("unexpected token in top-level statement")
  end
end

function parser.top_quote_expr(p, annotations)
  local start = ast.save(p)
  p:expect("rexpr")
  local expr = p:expr()
  p:expect("end")

  return ast.unspecialized.top.QuoteExpr {
    expr = expr,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.top_quote_stat(p, annotations)
  local start = ast.save(p)
  p:expect("rquote")
  local block = p:block()
  p:expect("end")

  return ast.unspecialized.top.QuoteStat {
    block = block,
    annotations = annotations,
    span = ast.span(start, p),
  }
end

function parser.top_expr(p)
  local annotations = p:annotations(false, true)

  if p:matches("rexpr") then
    return p:top_quote_expr(annotations)

  elseif p:matches("rquote") then
    return p:top_quote_stat(annotations)

  else
    p:error("unexpected token in top-level statement")
  end
end

function parser:entry_expr(lex)
  return parsing.Parse(self, lex, "top_expr")
end

function parser:entry_stat(lex)
  return parsing.Parse(self, lex, "top_stat")
end

return parser
