-- Copyright 2017 Stanford University
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

-- Typed AST Pretty Printer

local ast = require("regent/ast")
local std = require("regent/std_base")

local context = {}
context.__index = context

function context.new_global_scope()
  return setmetatable({
  }, context)
end

function context:new_render_scope()
  return setmetatable({
    indent = 0,
  }, context)
end

function context:new_indent_scope(increment)
  assert(increment >= 0)
  return setmetatable({
    indent = self.indent + increment,
  }, context)
end

local text = ast.make_factory("text")
text:leaf("Lines", {"lines"})
text:leaf("Indent", {"value"})
text:leaf("Line", {"value"})

local render = {}

function render.lines(cx, node)
  local result = terralib.newlist()
  for _, line in ipairs(node.lines) do
    result:insertall(render.top(cx, line))
  end
  return result
end

function render.indent(cx, node)
  cx = cx:new_indent_scope(1)
  return render.top(cx, node.value)
end

function render.line(cx, node)
  return terralib.newlist({string.rep("  ", cx.indent) .. tostring(node.value)})
end

function render.top(cx, node)
  if node:is(text.Lines) then
    return render.lines(cx, node)

  elseif node:is(text.Indent) then
    return render.indent(cx, node)

  elseif node:is(text.Line) then
    return render.line(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function render.entry(cx, node)
  if not cx then cx = context.new_render_scope() end
  return render.top(cx, node):concat("\n")
end

local pretty = {}

local function join(elts, loose)
  local space = (loose and " ") or ""
  local result = terralib.newlist()
  local line
  for _, a in ipairs(elts) do
    if not a then
      -- Pass
    elseif type(a) == "string" then
      if line then
        line = line .. space .. a
      else
        line = a
      end
    elseif a:is(text.Line) then
      if line then
        line = line .. space .. a.value
      else
        line = a.value
      end
    elseif a:is(text.Lines) then
      if line then
        result:insert(text.Line { value = line })
        line = nil
      end
      result:insert(a)
    else
      assert(false)
    end
  end

  if line then
    result:insert(text.Line { value = line })
  end
  if #result == 1 then
    return result[1]
  end
  return result
end

local function commas(elts)
  local result = terralib.newlist()
  local line
  for _, a in ipairs(elts) do
    if not a then
      -- Pass
    elseif type(a) == "string" then
      if line then
        line = line .. ", " .. a
      else
        if #result > 0 then result:insert(text.Line { value = "," }) end
        line = a
      end
    elseif a:is(text.Line) then
      if line then
        line = line .. ", " .. a.value
      else
        if #result > 0 then result:insert(text.Line { value = "," }) end
        line = a.value
      end
    elseif a:is(text.Lines) then
      if line then
        result:insert(text.Line { value = line .. "," })
        line = nil
      end
      result:insert(a)
    else
      assert(false)
    end
  end
  if line then
    if #result > 0 then result:insert(text.Line { value = "," }) end
    result:insert(text.Line { value = line })
  end
  if #result == 1 then
    return result[1]
  end
  if #result == 0 then
    return text.Line { value = "" }
  end
  return text.Lines { lines = result }
end

function pretty.expr_condition(cx, node)
  return join({
      join(node.conditions:map(tostring), true), "(", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_region_root(cx, node)
  if #node.fields == 1 then
    return join({pretty.expr(cx, node.region),
                 #(node.fields[1]) > 0 and ".",
                 #(node.fields[1]) > 0 and node.fields[1]:mkstring(".")})
  else
    return join({
        pretty.expr(cx, node.region), ".{",
        commas(node.fields:map(function(field_path) return field_path:mkstring(".") end)),
        "}"})
  end
end

function pretty.expr_id(cx, node)
  return text.Line { value = tostring(node.value) }
end

function pretty.expr_constant(cx, node)
  local value = node.value
  if type(node.value) == "string" then
    value = "\"" .. value:gsub("\n", "\\n"):gsub("\t", "\\t"):gsub("\"", "\\\"") .. "\""
  end
  return text.Line { value = tostring(value) }
end

function pretty.expr_function(cx, node)
  local name
  if std.is_task(node.value) then
    name = node.value:getname():mkstring(".")
  elseif terralib.isfunction(node.value) then
    name = node.value:getname()
  else
    name = tostring(node.value)
  end
  return text.Line { value = name }
end

function pretty.expr_field_access(cx, node)
  return join({pretty.expr(cx, node.value), "." .. node.field_name})
end

function pretty.expr_index_access(cx, node)
  return join({pretty.expr(cx, node.value), "[", pretty.expr(cx, node.index), "]"})
end

function pretty.expr_method_call(cx, node)
  return join({
    pretty.expr(cx, node.value),
    ":" .. tostring(node.method_name) .. "(",
    pretty.expr_list(cx, node.args), ")"})
end

function pretty.expr_call(cx, node)
  local args = terralib.newlist()
  args:insertall(node.args:map(function(arg) return pretty.expr(cx, arg) end))
  args:insertall(
    node.conditions:map(
      function(condition) return pretty.expr_condition(cx, condition) end))
  return join({pretty.expr(cx, node.fn), "(", commas(args) , ")"})
end

function pretty.expr_cast(cx, node)
  return join({
      pretty.expr(cx, node.fn), "(", pretty.expr(cx, node.arg), ")"})
end

function pretty.expr_ctor_list_field(cx, node)
  return pretty.expr(cx, node.value)
end

function pretty.expr_ctor_rec_field(cx, node)
  return join({node.name, "=", pretty.expr(cx, node.value)}, true)
end

function pretty.expr_ctor_field(cx, node)
  if node:is(ast.typed.expr.CtorListField) then
    return pretty.expr_ctor_list_field(cx, node)
  elseif node:is(ast.typed.expr.CtorRecField) then
    return pretty.expr_ctor_rec_field(cx, node)
  else
    assert(false)
  end
end

function pretty.expr_ctor(cx, node)
  return join({
      "{",
      commas(node.fields:map(
               function(field) return pretty.expr_ctor_field(cx, field) end)),
      "}"})
end

function pretty.expr_raw_context(cx, node)
  return text.Line { value = "__context()"}
end

function pretty.expr_raw_fields(cx, node)
  return join({"__fields(", pretty.expr(cx, node.region), ")"})
end

function pretty.expr_raw_physical(cx, node)
  return join({"__physical(", pretty.expr(cx, node.region), ")"})
end

function pretty.expr_raw_runtime(cx, node)
  return text.Line { value = "__runtime()"}
end

function pretty.expr_raw_value(cx, node)
  return join({"__raw(", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_isnull(cx, node)
  return join({"isnull(", pretty.expr(cx, node.pointer), ")"})
end

function pretty.expr_new(cx, node)
  return join({
      "new(",
      commas({tostring(node.pointer_type), pretty.expr(cx, node.region),
              node.extent and pretty.expr(cx, node.extent)}),
      ")"})
end

function pretty.expr_null(cx, node)
  return text.Line { value = "null(" .. tostring(node.pointer_type) .. ")" }
end

function pretty.expr_dynamic_cast(cx, node)
  return join({"dynamic_cast(", commas({tostring(node.expr_type), pretty.expr(cx, node.value)}), ")"})
end

function pretty.expr_static_cast(cx, node)
  return join({"static_cast(", commas({tostring(node.expr_type), pretty.expr(cx, node.value)}), ")"})
end

function pretty.expr_unsafe_cast(cx, node)
  return join({"unsafe_cast(", commas({tostring(node.expr_type), pretty.expr(cx, node.value)}), ")"})
end

function pretty.expr_ispace(cx, node)
  return join({
      "ispace(",
      commas({tostring(node.index_type), pretty.expr(cx, node.extent),
              node.start and pretty.expr(cx, node.start)}),
      ")"})
end

function pretty.expr_region(cx, node)
  return join({"region(", commas({pretty.expr(cx, node.ispace), tostring(node.fspace_type)}), ")"})
end

function pretty.expr_partition(cx, node)
  return join({
      "partition(",
      commas({tostring(node.disjointness),
              pretty.expr(cx, node.region), pretty.expr(cx, node.coloring)}),
      ")"})
end

function pretty.expr_partition_equal(cx, node)
  return join({
      "partition(",
      commas({"equal",
              pretty.expr(cx, node.region), pretty.expr(cx, node.colors)}),
      ")"})
end

function pretty.expr_partition_by_field(cx, node)
  return join({
      "partition(",
      commas({pretty.expr_region_root(cx, node.region),
              pretty.expr(cx, node.colors)}),
      ")"})
end

function pretty.expr_image(cx, node)
  return join({
      "image(",
      commas({pretty.expr(cx, node.parent),
              pretty.expr(cx, node.partition),
              pretty.expr_region_root(cx, node.region)}),
      ")"})
end

function pretty.expr_preimage(cx, node)
  return join({
      "preimage(",
      commas({pretty.expr(cx, node.parent),
              pretty.expr(cx, node.partition),
              pretty.expr_region_root(cx, node.region)}),
      ")"})
end

function pretty.expr_cross_product(cx, node)
  return join({
      "cross_product(", commas({pretty.expr_list(cx, node.args)}), ")"})
end

function pretty.expr_cross_product_array(cx, node)
  return join({
      "cross_product_array(",
      commas({pretty.expr(cx, node.lhs),
              tostring(node.disjointness),
              pretty.expr(cx, node.colorings)}),
      ")"})
end

function pretty.expr_list_slice_partition(cx, node)
  return join({
      "list_slice_partition(",
      commas({pretty.expr(cx, node.partition),
              pretty.expr(cx, node.indices)}),
      ")"})
end

function pretty.expr_list_duplicate_partition(cx, node)
  return join({
      "list_duplicate_partition(",
      commas({pretty.expr(cx, node.partition),
              pretty.expr(cx, node.indices)}),
      ")"})
end

function pretty.expr_list_slice_cross_product(cx, node)
  return join({
      "list_slice_cross_product(",
      commas({pretty.expr(cx, node.product),
              pretty.expr(cx, node.indices)}),
      ")"})
end

function pretty.expr_list_cross_product(cx, node)
  return join({
      "list_cross_product(",
      commas({pretty.expr(cx, node.lhs),
              pretty.expr(cx, node.rhs),
              tostring(node.shallow)}),
      ")"})
end

function pretty.expr_list_cross_product_complete(cx, node)
  return join({
      "list_cross_product_complete(",
      commas({pretty.expr(cx, node.lhs),
              pretty.expr(cx, node.product)}),
      ")"})
end

function pretty.expr_list_phase_barriers(cx, node)
  return join({
      "list_phase_barriers(", commas({pretty.expr(cx, node.product)}), ")"})
end

function pretty.expr_list_invert(cx, node)
  return join({
      "list_invert(",
      commas({pretty.expr(cx, node.rhs),
              pretty.expr(cx, node.product),
              pretty.expr(cx, node.barriers)}),
      ")"})
end

function pretty.expr_list_range(cx, node)
  return join({
      "list_range(",
      commas({pretty.expr(cx, node.start),
              pretty.expr(cx, node.stop)}),
      ")"})
end

function pretty.expr_list_ispace(cx, node)
  return join({
      "list_ispace(",
      commas({pretty.expr(cx, node.ispace)}),
      ")"})
end

function pretty.expr_phase_barrier(cx, node)
  return join({
      "phase_barrier(", commas({pretty.expr(cx, node.value)}), ")"})
end

function pretty.expr_dynamic_collective(cx, node)
  return join({
      "dynamic_collective(",
      commas({tostring(node.value_type), node.op, pretty.expr(cx, node.arrivals)}),
      ")"})
end

function pretty.expr_dynamic_collective_get_result(cx, node)
  return join({
      "dynamic_collective_get_result(",
      commas({pretty.expr(cx, node.value)}),
      ")"})
end

function pretty.expr_advance(cx, node)
  return join({
      "advance(", commas({pretty.expr(cx, node.value)}), ")"})
end

function pretty.expr_arrive(cx, node)
  return join({
      "arrive(",
      commas({pretty.expr(cx, node.barrier), node.value and pretty.expr(cx, node.value)}),
      ")"})
end

function pretty.expr_await(cx, node)
  return join({"await(", pretty.expr(cx, node.barrier), ")"})
end

function pretty.expr_copy(cx, node)
  return join({
      "copy(",
      commas({pretty.expr_region_root(cx, node.src),
              pretty.expr_region_root(cx, node.dst),
              node.op,
              unpack(node.conditions:map(
                function(condition) return pretty.expr_condition(cx, condition) end))}),
      ")"})
end

function pretty.expr_fill(cx, node)
  return join({
      "fill(",
      commas({pretty.expr_region_root(cx, node.dst),
              pretty.expr(cx, node.value),
              unpack(node.conditions:map(
                function(condition) return pretty.expr_condition(cx, condition) end))}),
      ")"})
end

function pretty.expr_acquire(cx, node)
  return join({
      "acquire(",
      commas({pretty.expr_region_root(cx, node.region),
              unpack(node.conditions:map(
                function(condition) return pretty.expr_condition(cx, condition) end))}),
      ")"})
end

function pretty.expr_release(cx, node)
  return join({
      "release(",
      commas({pretty.expr_region_root(cx, node.region),
              unpack(node.conditions:map(
                function(condition) return pretty.expr_condition(cx, condition) end))}),
      ")"})
end

function pretty.expr_attach_hdf5(cx, node)
  return join({
      "attach(",
      commas({"hdf5",
              pretty.expr_region_root(cx, node.region),
              pretty.expr(cx, node.filename),
              pretty.expr(cx, node.mode)}),
      ")"})
end

function pretty.expr_detach_hdf5(cx, node)
  return join({
      "detach(",
      commas({"hdf5",
              pretty.expr_region_root(cx, node.region)}),
      ")"})
end

function pretty.expr_allocate_scratch_fields(cx, node)
  return join({
      "allocate_scratch_fields(", pretty.expr_region_root(cx, node.region), ")"})
end

function pretty.expr_with_scratch_fields(cx, node)
  return join({
      "with_scratch_fields(",
      commas({pretty.expr_region_root(cx, node.region),
              pretty.expr(cx, node.field_ids)}),
      ")"})
end

function pretty.expr_unary(cx, node)
  local loose = node.op == "not"
  return join({"(", join({node.op, pretty.expr(cx, node.rhs)}, loose), ")"})
end

function pretty.expr_binary(cx, node)
  if node.op == "min" or node.op == "max" then
    return join({
        node.op, "(", commas({pretty.expr(cx, node.lhs), pretty.expr(cx, node.rhs)}), ")"})
  else
    local loose = node.op == "or" or node.op == "and"
    return join({
        "(", join({pretty.expr(cx, node.lhs), node.op, pretty.expr(cx, node.rhs)}, loose), ")"})
  end
end

function pretty.expr_deref(cx, node)
  return join({"(", "@", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_future(cx, node)
  return join({"__future(", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_future_get_result(cx, node)
  return join({"__future_get_result(", pretty.expr(cx, node.value), ")"})
end

function pretty.expr(cx, node)
  if not cx then cx = context.new_render_scope() end
  if node:is(ast.typed.expr.ID) then
    return pretty.expr_id(cx, node)

  elseif node:is(ast.typed.expr.Constant) then
    return pretty.expr_constant(cx, node)

  elseif node:is(ast.typed.expr.Function) then
    return pretty.expr_function(cx, node)

  elseif node:is(ast.typed.expr.FieldAccess) then
    return pretty.expr_field_access(cx, node)

  elseif node:is(ast.typed.expr.IndexAccess) then
    return pretty.expr_index_access(cx, node)

  elseif node:is(ast.typed.expr.MethodCall) then
    return pretty.expr_method_call(cx, node)

  elseif node:is(ast.typed.expr.Call) then
    return pretty.expr_call(cx, node)

  elseif node:is(ast.typed.expr.Cast) then
    return pretty.expr_cast(cx, node)

  elseif node:is(ast.typed.expr.Ctor) then
    return pretty.expr_ctor(cx, node)

  elseif node:is(ast.typed.expr.RawContext) then
    return pretty.expr_raw_context(cx, node)

  elseif node:is(ast.typed.expr.RawFields) then
    return pretty.expr_raw_fields(cx, node)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return pretty.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return pretty.expr_raw_runtime(cx, node)

  elseif node:is(ast.typed.expr.RawValue) then
    return pretty.expr_raw_value(cx, node)

  elseif node:is(ast.typed.expr.Isnull) then
    return pretty.expr_isnull(cx, node)

  elseif node:is(ast.typed.expr.New) then
    return pretty.expr_new(cx, node)

  elseif node:is(ast.typed.expr.Null) then
    return pretty.expr_null(cx, node)

  elseif node:is(ast.typed.expr.DynamicCast) then
    return pretty.expr_dynamic_cast(cx, node)

  elseif node:is(ast.typed.expr.StaticCast) then
    return pretty.expr_static_cast(cx, node)

  elseif node:is(ast.typed.expr.UnsafeCast) then
    return pretty.expr_unsafe_cast(cx, node)

  elseif node:is(ast.typed.expr.Ispace) then
    return pretty.expr_ispace(cx, node)

  elseif node:is(ast.typed.expr.Region) then
    return pretty.expr_region(cx, node)

  elseif node:is(ast.typed.expr.Partition) then
    return pretty.expr_partition(cx, node)

  elseif node:is(ast.typed.expr.PartitionEqual) then
    return pretty.expr_partition_equal(cx, node)

  elseif node:is(ast.typed.expr.PartitionByField) then
    return pretty.expr_partition_by_field(cx, node)

  elseif node:is(ast.typed.expr.Image) then
    return pretty.expr_image(cx, node)

  elseif node:is(ast.typed.expr.Preimage) then
    return pretty.expr_preimage(cx, node)

  elseif node:is(ast.typed.expr.CrossProduct) then
    return pretty.expr_cross_product(cx, node)

  elseif node:is(ast.typed.expr.CrossProductArray) then
    return pretty.expr_cross_product_array(cx, node)

  elseif node:is(ast.typed.expr.ListSlicePartition) then
    return pretty.expr_list_slice_partition(cx, node)

  elseif node:is(ast.typed.expr.ListDuplicatePartition) then
    return pretty.expr_list_duplicate_partition(cx, node)

  elseif node:is(ast.typed.expr.ListSliceCrossProduct) then
    return pretty.expr_list_slice_cross_product(cx, node)

  elseif node:is(ast.typed.expr.ListCrossProduct) then
    return pretty.expr_list_cross_product(cx, node)

  elseif node:is(ast.typed.expr.ListCrossProductComplete) then
    return pretty.expr_list_cross_product_complete(cx, node)

  elseif node:is(ast.typed.expr.ListPhaseBarriers) then
    return pretty.expr_list_phase_barriers(cx, node)

  elseif node:is(ast.typed.expr.ListInvert) then
    return pretty.expr_list_invert(cx, node)

  elseif node:is(ast.typed.expr.ListRange) then
    return pretty.expr_list_range(cx, node)

  elseif node:is(ast.typed.expr.ListIspace) then
    return pretty.expr_list_ispace(cx, node)

  elseif node:is(ast.typed.expr.PhaseBarrier) then
    return pretty.expr_phase_barrier(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollective) then
    return pretty.expr_dynamic_collective(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollectiveGetResult) then
    return pretty.expr_dynamic_collective_get_result(cx, node)

  elseif node:is(ast.typed.expr.Advance) then
    return pretty.expr_advance(cx, node)

  elseif node:is(ast.typed.expr.Arrive) then
    return pretty.expr_arrive(cx, node)

  elseif node:is(ast.typed.expr.Await) then
    return pretty.expr_await(cx, node)

  elseif node:is(ast.typed.expr.Copy) then
    return pretty.expr_copy(cx, node)

  elseif node:is(ast.typed.expr.Fill) then
    return pretty.expr_fill(cx, node)

  elseif node:is(ast.typed.expr.Acquire) then
    return pretty.expr_acquire(cx, node)

  elseif node:is(ast.typed.expr.Release) then
    return pretty.expr_release(cx, node)

  elseif node:is(ast.typed.expr.AttachHDF5) then
    return pretty.expr_attach_hdf5(cx, node)

  elseif node:is(ast.typed.expr.DetachHDF5) then
    return pretty.expr_detach_hdf5(cx, node)

  elseif node:is(ast.typed.expr.AllocateScratchFields) then
    return pretty.expr_allocate_scratch_fields(cx, node)

  elseif node:is(ast.typed.expr.WithScratchFields) then
    return pretty.expr_with_scratch_fields(cx, node)

  elseif node:is(ast.typed.expr.Unary) then
    return pretty.expr_unary(cx, node)

  elseif node:is(ast.typed.expr.Binary) then
    return pretty.expr_binary(cx, node)

  elseif node:is(ast.typed.expr.Deref) then
    return pretty.expr_deref(cx, node)

  elseif node:is(ast.typed.expr.Future) then
    return pretty.expr_future(cx, node)

  elseif node:is(ast.typed.expr.FutureGetResult) then
    return pretty.expr_future_get_result(cx, node)

  elseif node:is(ast.typed.expr.ParallelizerConstraint) then
    return pretty.expr_binary(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function pretty.expr_list(cx, node)
  return commas(node:map(function(expr) return pretty.expr(cx, expr) end))
end

function pretty.block(cx, node)
  return text.Indent {
    value = text.Lines {
      lines = node.stats:map(function(stat) return pretty.stat(cx, stat) end),
    }
  }
end

function pretty.stat_if(cx, node)
  local result = terralib.newlist()
  result:insert(join({"if", pretty.expr(cx, node.cond), "then"}, true))
  result:insert(pretty.block(cx, node.then_block))
  for _, elseif_block in ipairs(node.elseif_blocks) do
    result:insert(join({"elseif", pretty.expr(cx, elseif_block.cond), "then"}, true))
    result:insert(pretty.block(cx, elseif_block.block))
  end
  result:insert(text.Line { value = "else" })
  result:insert(pretty.block(cx, node.else_block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_while(cx, node)
  local result = terralib.newlist()
  result:insert(join({"while", pretty.expr(cx, node.cond), "do"}, true))
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_for_num(cx, node)
  local result = terralib.newlist()
  result:insert(join({"for", tostring(node.symbol),
                      ":", tostring(node.symbol:gettype()),
                      "=", pretty.expr_list(cx, node.values), "do"}, true))
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_for_list(cx, node)
  local result = terralib.newlist()
  result:insert(join({"for", tostring(node.symbol),
                      ":", tostring(node.symbol:gettype()),
                      "in", pretty.expr(cx, node.value), "do"}, true))
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_for_list_vectorized(cx, node)
  local result = terralib.newlist()
  result:insert(join({"for", tostring(node.symbol),
                      ":", tostring(node.symbol:gettype()),
                      "in", pretty.expr(cx, node.value), "do -- vectorized"}, true))
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_repeat(cx, node)
  local result = terralib.newlist()
  result:insert(text.Line { value = "repeat" })
  result:insert(pretty.block(cx, node.block))
  result:insert(join({"until", pretty.expr(cx, node.until_cond)}, true))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_must_epoch(cx, node)
  local result = terralib.newlist()
  result:insert(text.Line { value = "must_epoch" })
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_block(cx, node)
  local result = terralib.newlist()
  result:insert(text.Line { value = "do" })
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_index_launch_num(cx, node)
  local result = terralib.newlist()
  result:insert(join({"for", tostring(node.symbol), "=", pretty.expr_list(cx, node.values), "do -- index launch"}, true))
  local call = pretty.expr(cx, node.call)
  if node.reduce_op then
    call = join({pretty.expr(cx, node.reduce_lhs), node.reduce_op .. "=", call}, true)
  end
  result:insert(text.Indent { value = call })
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_index_launch_list(cx, node)
  local result = terralib.newlist()
  result:insert(join({"for", tostring(node.symbol), "in", pretty.expr(cx, node.value), "do -- index launch"}, true))
  local call = pretty.expr(cx, node.call)
  if node.reduce_op then
    call = join({pretty.expr(cx, node.reduce_lhs), node.reduce_op .. "=", call}, true)
  end
  result:insert(text.Indent { value = call })
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_var(cx, node)
  local symbols = commas(node.symbols:map(function(symbol) return tostring(symbol) end))
  local types = commas(node.types:map(function(type) return tostring(type) end))
  local assign = #node.values > 0 and "="
  return join({"var", symbols, ":", types, assign, pretty.expr_list(cx, node.values)}, true)
end

function pretty.stat_var_unpack(cx, node)
  local symbols = commas(node.symbols:map(function(symbol) return tostring(symbol) end))
  local fields = commas(node.fields:map(function(field) return field end))
  return join({"var", "{", symbols, "=", fields, "}", "=", pretty.expr(cx, node.value)}, true)
end

function pretty.stat_return(cx, node)
  return join({"return", node.value and pretty.expr(cx, node.value)}, true)
end

function pretty.stat_break(cx, node)
  return text.Line { value = "break" }
end

function pretty.stat_assignment(cx, node)
  return join({pretty.expr_list(cx, node.lhs), "=", pretty.expr_list(cx, node.rhs)}, true)
end

function pretty.stat_reduce(cx, node)
  return join({pretty.expr_list(cx, node.lhs), node.op .. "=", pretty.expr_list(cx, node.rhs)}, true)
end

function pretty.stat_expr(cx, node)
  return pretty.expr(cx, node.expr)
end

function pretty.stat_begin_trace(cx, node)
  return join({"__begin_trace(", pretty.expr(cx, node.trace_id), ")"})
end

function pretty.stat_end_trace(cx, node)
  return join({"__end_trace(", pretty.expr(cx, node.trace_id), ")"})
end

function pretty.stat_map_regions(cx, node)
  return join({
    "__map_regions(",
    commas(node.region_types:map(function(region) return tostring(region) end)),
    ")"})
end

function pretty.stat_unmap_regions(cx, node)
  return join({
    "__unmap_regions(",
    commas(node.region_types:map(function(region) return tostring(region) end)),
    ")"})
end

function pretty.stat_raw_delete(cx, node)
  return join({
    "__delete(",
    pretty.expr(cx, node.value),
    ")"})
end

function pretty.stat_parallelize_with(cx, node)
  local result = terralib.newlist()
  result:insert(join({"__parallelize_with ",
    commas(node.hints:map(function(hint) return pretty.expr(cx, hint) end)),
    " do"}))
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat(cx, node)
  if not cx then cx = context.new_global_scope() end
  if node:is(ast.typed.stat.If) then
    return pretty.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    return pretty.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    return pretty.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForList) then
    return pretty.stat_for_list(cx, node)

  elseif node:is(ast.typed.stat.ForListVectorized) then
    return pretty.stat_for_list_vectorized(cx, node)

  elseif node:is(ast.typed.stat.Repeat) then
    return pretty.stat_repeat(cx, node)

  elseif node:is(ast.typed.stat.MustEpoch) then
    return pretty.stat_must_epoch(cx, node)

  elseif node:is(ast.typed.stat.Block) then
    return pretty.stat_block(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchNum) then
    return pretty.stat_index_launch_num(cx, node)

  elseif node:is(ast.typed.stat.IndexLaunchList) then
    return pretty.stat_index_launch_list(cx, node)

  elseif node:is(ast.typed.stat.Var) then
    return pretty.stat_var(cx, node)

  elseif node:is(ast.typed.stat.VarUnpack) then
    return pretty.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.stat.Return) then
    return pretty.stat_return(cx, node)

  elseif node:is(ast.typed.stat.Break) then
    return pretty.stat_break(cx, node)

  elseif node:is(ast.typed.stat.Assignment) then
    return pretty.stat_assignment(cx, node)

  elseif node:is(ast.typed.stat.Reduce) then
    return pretty.stat_reduce(cx, node)

  elseif node:is(ast.typed.stat.Expr) then
    return pretty.stat_expr(cx, node)

  elseif node:is(ast.typed.stat.BeginTrace) then
    return pretty.stat_begin_trace(cx, node)

  elseif node:is(ast.typed.stat.EndTrace) then
    return pretty.stat_end_trace(cx, node)

  elseif node:is(ast.typed.stat.MapRegions) then
    return pretty.stat_map_regions(cx, node)

  elseif node:is(ast.typed.stat.UnmapRegions) then
    return pretty.stat_unmap_regions(cx, node)

  elseif node:is(ast.typed.stat.RawDelete) then
    return pretty.stat_raw_delete(cx, node)

  elseif node:is(ast.typed.stat.ParallelizeWith) then
    return pretty.stat_parallelize_with(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function pretty.top_task_param(cx, node)
  return tostring(node.symbol) .. " : " .. tostring(node.param_type)
end

function pretty.top_task_privileges(cx, node)
  local result = terralib.newlist()
  for _, privileges in ipairs(node) do
    for _, privilege in ipairs(privileges) do
      result:insert(join({
        tostring(privilege.privilege),
        "(",
        terralib.newlist({
            tostring(privilege.region),
            unpack(privilege.field_path)}):concat("."),
        ")"}))
    end
  end
  return result
end

function pretty.top_task_coherence_modes(cx, node, mapping)
  local result = terralib.newlist()
  for region, coherence_modes in node:items() do
    for field_path, coherence_mode in coherence_modes:items() do
      result:insert(join({
        tostring(coherence_mode),
        "(",
        terralib.newlist({
            mapping[region],
            unpack(field_path)}):concat("."),
        ")"}))
    end
  end
  return result
end

function pretty.top_task_flags(cx, node, mapping)
  local result = terralib.newlist()
  for region, flags in node:items() do
    for field_path, flag in flags:items() do
      flag:map(function(flag)
        result:insert(join({
          tostring(flag),
          "(",
          terralib.newlist({
              mapping[region],
              unpack(field_path)}):concat("."),
          ")"}))
      end)
    end
  end
  return result
end

function pretty.top_task_conditions(cx, node)
  local result = terralib.newlist()
  for condition, values in pairs(node) do
    for _, value in pairs(values) do
      result:insert(join({tostring(condition), "(", tostring(value), ")" }))
    end
  end
  return result
end

function pretty.top_task_constraints(cx, node)
  if not node then return terralib.newlist() end
  return node:map(
    function(constraint)
      return join({tostring(constraint.lhs), tostring(constraint.op),
                   tostring(constraint.rhs)},
        true)
    end)
end

function pretty.task_config_options(cx, node)
  return terralib.newlist({
      join({"leaf (", tostring(node.leaf), ")"}),
      join({"inner (", tostring(node.inner), ")"}),
      join({"idempotent (", tostring(node.idempotent), ")"}),
  })
end

function pretty.top_task(cx, node)
  local name = node.name:concat(".")
  local params = commas(node.params:map(
                          function(param) return pretty.top_task_param(cx, param) end))
  local return_type = ""
  if node.return_type ~= terralib.types.unit then
    return_type = " : " .. tostring(node.return_type)
  end
  local mapping = {}
  node.privileges:map(function(privileges)
    privileges:map(function(privilege)
      mapping[privilege.region:gettype()] = tostring(privilege.region)
    end)
  end)
  local meta = terralib.newlist()
  meta:insertall(pretty.top_task_privileges(cx, node.privileges))
  meta:insertall(pretty.top_task_coherence_modes(cx, node.coherence_modes, mapping))
  meta:insertall(pretty.top_task_flags(cx, node.flags, mapping))
  meta:insertall(pretty.top_task_conditions(cx, node.conditions))
  meta:insertall(pretty.top_task_constraints(cx, node.constraints))
  local config_options = pretty.task_config_options(cx, node.config_options)

  local lines = terralib.newlist()
  lines:insert(join({"task " .. name, "(", params, ")", return_type }))
  lines:insert(join({"-- ", commas(config_options) }))
  if #meta > 0 then
    lines:insert(text.Line { value = "where" })
    lines:insert(text.Indent { value = commas(meta) })
    lines:insert(text.Line { value = "do" })
  end
  lines:insert(pretty.block(cx, node.body))
  lines:insert(text.Line { value = "end" })

  return text.Lines { lines = lines }
end

function pretty.top_fspace(cx, node)
  -- TODO: Pretty-print fspaces
  return text.Line { value = "" }
end

function pretty.top(cx, node)
  if node:is(ast.typed.top.Task) then
    return pretty.top_task(cx, node)

  elseif node:is(ast.typed.top.Fspace) then
    return pretty.top_fspace(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function pretty.entry(node)
  local cx = context.new_global_scope()
  return render.entry(cx:new_render_scope(), pretty.top(cx, node))
end

pretty.render = render

return pretty

