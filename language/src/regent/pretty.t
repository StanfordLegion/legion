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

-- Typed AST Pretty Printer

local ast = require("regent/ast")
local std = require("regent/std_base")
local symbol_table = require("regent/symbol_table")

local pretty_context = {}

function pretty_context:__index (field)
  local value = pretty_context [field]
  if value ~= nil then
    return value
  end
  error ("pretty_context has no field '" .. field .. "' (in lookup)", 2)
end

function pretty_context:__newindex (field, value)
  error ("pretty_context has no field '" .. field .. "' (in assignment)", 2)
end

function pretty_context.new_global_scope()
  return setmetatable({
    name_map = symbol_table.new_global_scope({}),
    visible_names = symbol_table.new_global_scope({}),
  }, pretty_context)
end

function pretty_context:new_local_scope()
  return setmetatable({
    name_map = self.name_map:new_local_scope(),
    visible_names = self.visible_names:new_local_scope(),
  }, pretty_context)
end

function pretty_context:print_var(v)
  local name = self.name_map:safe_lookup(v)
  return name or tostring(v)
end

function pretty_context:record_var(node, v)
  if std.config['debug'] or not v:hasname() then
    return
  end
  if self.visible_names:safe_lookup(v:getname()) then
    local unique_name = '$' .. v:getname() .. '#' .. tostring(v.symbol_id)
    self.name_map:insert(node, v, unique_name)
  else
    self.visible_names:insert(node, v:getname(), true)
  end
end

local render_context = {}

function render_context:__index (field)
  local value = render_context [field]
  if value ~= nil then
    return value
  end
  error ("render_context has no field '" .. field .. "' (in lookup)", 2)
end

function render_context:__newindex (field, value)
  error ("render_context has no field '" .. field .. "' (in assignment)", 2)
end

function render_context.new_render_scope()
  return setmetatable({
    indent = 0,
  }, render_context)
end

function render_context:new_indent_scope(increment)
  assert(increment >= 0)
  return setmetatable({
    indent = self.indent + increment,
  }, render_context)
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

function pretty.annotations(cx, node)
  local fields = node:get_fields()
  local demand = terralib.newlist()
  local forbid = terralib.newlist()
  for k, v in fields:items() do
    if v:is(ast.annotation.Demand) then
      demand:insert("__" .. k)
    end
    if v:is(ast.annotation.Forbid) then
      forbid:insert("__" .. k)
    end
  end

  local result = terralib.newlist()
  if #demand > 0 then
    result:insert(join({"__demand(", commas(demand), ")"}))
  end
  if #forbid > 0 then
    result:insert(join({"__forbid(", commas(forbid), ")"}))
  end
  return text.Lines { lines = result }
end

function pretty.metadata(cx, node)
  if node:is(ast.metadata.Loop) then
    local result = terralib.newlist()
    result:insertall({"-- parallelizable: ", tostring(node.parallelizable)})
    if node.reductions and #node.reductions > 0 then
      result:insert(", reductions: ")
      result:insert(commas(node.reductions:map(function(reduction)
        return tostring(reduction)
      end)))
    end
    return text.Lines { lines = terralib.newlist({ join(result) }) }
  elseif node:is(ast.metadata.Stat) then
    local centers =
      (node.centers and "{" ..  node.centers:map_list(function(k, _) return tostring(k) end):concat(",") .. "}") or
      tostring(node.centers)
    return text.Lines {
      lines = terralib.newlist({
        join({"-- ", "centers: ", centers, ", scalar: ", tostring(node.scalar)})}),
    }
  else
    assert(false)
  end
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
  return text.Line { value = cx:print_var(node.value) }
end

function pretty.expr_constant(cx, node)
  local value = node.value
  if type(node.value) == "string" then
    value = "\"" .. value:gsub("\n", "\\n"):gsub("\t", "\\t"):gsub("\"", "\\\"") .. "\""
  elseif type(value) == "cdata" and node.expr_type:isfloat() then
    value = tonumber(value)
  end
  return text.Line { value = tostring(value) }
end

function pretty.expr_global(cx, node)
  return text.Line { value = tostring(node.value) }
end

function pretty.expr_function(cx, node)
  local name
  if std.is_task(node.value) then
    name = node.value:get_name():mkstring(".")
  elseif terralib.isfunction(node.value) then
    name = node.value:getname()
  elseif node.value == _G['array'] or node.value == _G['arrayof'] then
    name = 'array'
  elseif node.value == _G['vector'] or node.value == _G['vectorof'] then
    name = 'vector'
  elseif node.value == _G['tuple'] then
    name = 'tuple'
  elseif regentlib.is_math_fn(node.value) then
    name = node.value:printpretty()
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
  if node.predicate then
    args:insert(join({"predicate=", pretty.expr(cx, node.predicate)}))
  end
  if node.predicate_else_value then
    args:insert(join({"predicate_else_value=", pretty.expr(cx, node.predicate_else_value)}))
  end
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
  return join({"__fields(", pretty.expr_region_root(cx, node.region), ")"})
end

function pretty.expr_raw_future(cx, node)
  return join({"__future(", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_raw_physical(cx, node)
  return join({"__physical(", pretty.expr_region_root(cx, node.region), ")"})
end

function pretty.expr_raw_runtime(cx, node)
  return text.Line { value = "__runtime()"}
end

function pretty.expr_raw_task(cx, node)
  return text.Line { value = "__task()"}
end

function pretty.expr_raw_value(cx, node)
  return join({"__raw(", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_isnull(cx, node)
  return join({"isnull(", pretty.expr(cx, node.pointer), ")"})
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
  local args = terralib.newlist()
  args:insert(tostring(node.disjointness))
  args:insert(pretty.expr(cx, node.region))
  args:insert(pretty.expr(cx, node.coloring))
  if node.colors then args:insert(pretty.expr(cx, node.colors)) end
  return join({"partition(", commas(args), ")"})
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

function pretty.expr_partition_by_restriction(cx, node)
  return join({
      "restrict(",
      commas({node.disjointness and tostring(node.disjointness),
	      pretty.expr(cx, node.region),
              pretty.expr(cx, node.transform),
              pretty.expr(cx, node.extent),
              pretty.expr(cx, node.colors)}),
      ")"})
end

function pretty.expr_image(cx, node)
  return join({
      "image(",
      commas({node.disjointness and tostring(node.disjointness),
	      pretty.expr(cx, node.parent),
              pretty.expr(cx, node.partition),
              pretty.expr_region_root(cx, node.region)}),
      ")"})
end

function pretty.expr_preimage(cx, node)
  return join({
      "preimage(",
      commas({node.disjointness and tostring(node.disjointness),
	      pretty.expr(cx, node.parent),
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

function pretty.expr_list_from_element(cx, node)
  return join({
      "list_from_element(",
      commas({pretty.expr(cx, node.list),
             pretty.expr(cx, node.value)}),
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

function pretty.expr_adjust(cx, node)
  return join({
      "adjust(",
      commas({pretty.expr(cx, node.barrier), pretty.expr(cx, node.value)}),
      ")"})
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
              pretty.expr(cx, node.mode),
              node.field_map and pretty.expr(cx, node.field_map)}),
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

function pretty.expr_address_of(cx, node)
  return join({"(", "&", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_future(cx, node)
  return join({"__future(", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_future_get_result(cx, node)
  return join({"__future_get_result(", pretty.expr(cx, node.value), ")"})
end

function pretty.expr_import_ispace(cx, node)
  return join({
      "__import_ispace(",
      commas({tostring(node.expr_type.index_type), pretty.expr(cx, node.value)}),
      ")"})
end

function pretty.expr_import_region(cx, node)
  return join({
      "__import_region(",
      commas({pretty.expr(cx, node.ispace), tostring(node.expr_type:fspace()),
              pretty.expr(cx, node.value), pretty.expr(cx, node.field_ids)}),
      ")"})
end

function pretty.expr_import_partition(cx, node)
  return join({
      "__import_partition(",
      commas({tostring(node.expr_type.disjointness), pretty.expr(cx, node.region),
              pretty.expr(cx, node.colors), pretty.expr(cx, node.value)}),
      ")"})
end

function pretty.expr_import_cross_product(cx, node)
  return join({
      "__import_cross_product(",
      commas({pretty.expr_list(cx, node.partitions),
              pretty.expr(cx, node.colors),
              pretty.expr(cx, node.value)}),
      ")"})
end

function pretty.expr_projection(cx, node)
  return join({
    pretty.expr(cx, node.region),
    ".{",
    node.field_mapping:map(function(entry)
      if #entry[2] == 0 or entry[1] == entry[2] then
        return entry[1]:mkstring(".")
      else
        return entry[2]:mkstring(".") .. "=" .. entry[1]:mkstring(".")
      end
    end):concat(","),
    "}"})
end

function pretty.expr(cx, node)
  if node:is(ast.typed.expr.ID) then
    return pretty.expr_id(cx, node)

  elseif node:is(ast.typed.expr.Constant) then
    return pretty.expr_constant(cx, node)

  elseif node:is(ast.typed.expr.Global) then
    return pretty.expr_global(cx, node)

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

  elseif node:is(ast.typed.expr.RawFuture) then
    return pretty.expr_raw_future(cx, node)

  elseif node:is(ast.typed.expr.RawPhysical) then
    return pretty.expr_raw_physical(cx, node)

  elseif node:is(ast.typed.expr.RawRuntime) then
    return pretty.expr_raw_runtime(cx, node)

  elseif node:is(ast.typed.expr.RawTask) then
    return pretty.expr_raw_task(cx, node)

  elseif node:is(ast.typed.expr.RawValue) then
    return pretty.expr_raw_value(cx, node)

  elseif node:is(ast.typed.expr.Isnull) then
    return pretty.expr_isnull(cx, node)

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

  elseif node:is(ast.typed.expr.PartitionByRestriction) then
    return pretty.expr_partition_by_restriction(cx, node)

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

  elseif node:is(ast.typed.expr.ListFromElement) then
    return pretty.expr_list_from_element(cx, node)

  elseif node:is(ast.typed.expr.PhaseBarrier) then
    return pretty.expr_phase_barrier(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollective) then
    return pretty.expr_dynamic_collective(cx, node)

  elseif node:is(ast.typed.expr.DynamicCollectiveGetResult) then
    return pretty.expr_dynamic_collective_get_result(cx, node)

  elseif node:is(ast.typed.expr.Advance) then
    return pretty.expr_advance(cx, node)

  elseif node:is(ast.typed.expr.Adjust) then
    return pretty.expr_adjust(cx, node)

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

  elseif node:is(ast.typed.expr.AddressOf) then
    return pretty.expr_address_of(cx, node)

  elseif node:is(ast.typed.expr.Future) then
    return pretty.expr_future(cx, node)

  elseif node:is(ast.typed.expr.FutureGetResult) then
    return pretty.expr_future_get_result(cx, node)

  elseif node:is(ast.typed.expr.ParallelizerConstraint) then
    return pretty.expr_binary(cx, node)

  elseif node:is(ast.typed.expr.ImportIspace) then
    return pretty.expr_import_ispace(cx, node)

  elseif node:is(ast.typed.expr.ImportRegion) then
    return pretty.expr_import_region(cx, node)

  elseif node:is(ast.typed.expr.ImportPartition) then
    return pretty.expr_import_partition(cx, node)

  elseif node:is(ast.typed.expr.ImportCrossProduct) then
    return pretty.expr_import_cross_product(cx, node)

  elseif node:is(ast.typed.expr.Projection) then
    return pretty.expr_projection(cx, node)

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
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({"if", pretty.expr(cx, node.cond), "then"}, true))
  result:insert(pretty.block(cx:new_local_scope(), node.then_block))
  for _, elseif_block in ipairs(node.elseif_blocks) do
    result:insert(join({"elseif", pretty.expr(cx, elseif_block.cond), "then"}, true))
    result:insert(pretty.block(cx:new_local_scope(), elseif_block.block))
  end
  result:insert(text.Line { value = "else" })
  result:insert(pretty.block(cx:new_local_scope(), node.else_block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_while(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({"while", pretty.expr(cx, node.cond), "do"}, true))
  result:insert(pretty.block(cx:new_local_scope(), node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_for_num(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  if std.config["pretty-verbose"] and node.metadata then
    result:insert(pretty.metadata(cx, node.metadata))
  end
  local values = pretty.expr_list(cx, node.values)
  local cx = cx:new_local_scope()
  cx:record_var(node, node.symbol)
  result:insert(join({"for", cx:print_var(node.symbol),
                      ":", tostring(node.symbol:gettype()),
                      "=", values, "do"}, true))
  local cx = cx:new_local_scope()
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_for_num_vectorized(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  local values = pretty.expr_list(cx, node.values)
  local cx = cx:new_local_scope()
  cx:record_var(node, node.symbol)
  result:insert(join({"for", cx:print_var(node.symbol),
                      ":", tostring(node.symbol:gettype()),
                      "=", values, "do -- vectorized"}, true))
  local cx = cx:new_local_scope()
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_for_list(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  if std.config["pretty-verbose"] and node.metadata then
    result:insert(pretty.metadata(cx, node.metadata))
  end
  local value = pretty.expr(cx, node.value)
  local cx = cx:new_local_scope()
  cx:record_var(node, node.symbol)
  result:insert(join({"for", cx:print_var(node.symbol),
                      ":", tostring(node.symbol:gettype()),
                      "in", value, "do"}, true))
  local cx = cx:new_local_scope()
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_for_list_vectorized(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  local value = pretty.expr(cx, node.value)
  local cx = cx:new_local_scope()
  cx:record_var(node, node.symbol)
  result:insert(join({"for", cx:print_var(node.symbol),
                      ":", tostring(node.symbol:gettype()),
                      "in", value, "do -- vectorized"}, true))
  local cx = cx:new_local_scope()
  result:insert(pretty.block(cx, node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_repeat(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(text.Line { value = "repeat" })
  result:insert(pretty.block(cx:new_local_scope(), node.block))
  result:insert(join({"until", pretty.expr(cx, node.until_cond)}, true))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_must_epoch(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(text.Line { value = "must_epoch" })
  result:insert(pretty.block(cx:new_local_scope(), node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_block(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(text.Line { value = "do" })
  result:insert(pretty.block(cx:new_local_scope(), node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_index_launch_num(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  local values = pretty.expr_list(cx, node.values)
  local cx = cx:new_local_scope()
  cx:record_var(node, node.symbol)
  local ctl = "(linear time launch)"
  if node.is_constant_time then
    ctl = "(constant time launch)"
  end
  result:insert(join({"for", cx:print_var(node.symbol), "=", values,
                      "do -- index launch", ctl}, true))
  local call = pretty.expr(cx, node.call)
  if node.reduce_op then
    call = join({pretty.expr(cx, node.reduce_lhs), node.reduce_op .. "=", call}, true)
  end
  node.preamble:map(function(stat)
    result:insert(text.Indent { value = pretty.stat(cx, stat) })
  end)
  result:insert(text.Indent { value = call })
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_index_launch_list(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  local value = pretty.expr(cx, node.value)
  local cx = cx:new_local_scope()
  cx:record_var(node, node.symbol)
  local ctl = "(linear time launch)"
  if node.is_constant_time then
    ctl = "(constant time launch)"
  end
  result:insert(join({"for", cx:print_var(node.symbol), "in", value,
                      "do -- index launch", ctl}, true))
  local call = pretty.expr(cx, node.call)
  if node.reduce_op then
    call = join({pretty.expr(cx, node.reduce_lhs), node.reduce_op .. "=", call}, true)
  end
  node.preamble:map(function(stat)
    result:insert(text.Indent { value = pretty.stat(cx, stat) })
  end)
  result:insert(text.Indent { value = call })
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_var(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  local value = node.value and pretty.expr(cx, node.value)
  cx:record_var(node, node.symbol)
  local decl = join({cx:print_var(node.symbol), ":", tostring(node.type)}, true)
  if value then
    result:insert(join({"var", decl, "=", value}, true))
  else
    result:insert(join({"var", decl}, true))
  end
  return text.Lines { lines = result }
end

function pretty.stat_var_unpack(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  local value = pretty.expr(cx, node.value)
  for _, symbol in ipairs(node.symbols) do
    cx:record_var(node, symbol)
  end
  local symbols = commas(node.symbols:map(function(symbol) return cx:print_var(symbol) end))
  local fields = commas(node.fields:map(function(field) return field end))
  result:insert(join({"var", "{", symbols, "=", fields, "}", "=", value}, true))
  return text.Lines { lines = result }
end

function pretty.stat_return(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({"return", node.value and pretty.expr(cx, node.value)}, true))
  return text.Lines { lines = result }
end

function pretty.stat_break(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(text.Line { value = "break" })
  return text.Lines { lines = result }
end

function pretty.stat_assignment(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  if std.config["pretty-verbose"] and node.metadata then
    result:insert(pretty.metadata(cx, node.metadata))
  end
  result:insert(join({pretty.expr(cx, node.lhs), "=", pretty.expr(cx, node.rhs)}, true))
  return text.Lines { lines = result }
end

function pretty.stat_reduce(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  if std.config["pretty-verbose"] and node.metadata then
    result:insert(pretty.metadata(cx, node.metadata))
  end
  result:insert(join({pretty.expr(cx, node.lhs), node.op .. "=", pretty.expr(cx, node.rhs)}, true))
  return text.Lines { lines = result }
end

function pretty.stat_expr(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(pretty.expr(cx, node.expr))
  return text.Lines { lines = result }
end

function pretty.stat_begin_trace(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({"__begin_trace(", pretty.expr(cx, node.trace_id), ")"}))
  return text.Lines { lines = result }
end

function pretty.stat_end_trace(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({"__end_trace(", pretty.expr(cx, node.trace_id), ")"}))
  return text.Lines { lines = result }
end

function pretty.stat_map_regions(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({
    "__map_regions(",
    commas(node.region_types:map(function(region) return tostring(region) end)),
    ")"}))
  return text.Lines { lines = result }
end

function pretty.stat_unmap_regions(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({
    "__unmap_regions(",
    commas(node.region_types:map(function(region) return tostring(region) end)),
    ")"}))
  return text.Lines { lines = result }
end

function pretty.stat_raw_delete(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({
    "__delete(",
    pretty.expr(cx, node.value),
    ")"}))
  return text.Lines { lines = result }
end

function pretty.stat_fence(cx, node)
  local args = terralib.newlist({tostring(node.kind)})
  if node.blocking then args:insert("__block") end

  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({
    "__fence(",
    commas(args),
    ")"}))
  return text.Lines { lines = result }
end

function pretty.stat_parallelize_with(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({"__parallelize_with ",
    commas(node.hints:map(function(hint) return pretty.expr(cx, hint) end)),
    " do"}))
  result:insert(pretty.block(cx:new_local_scope(), node.block))
  result:insert(text.Line { value = "end" })
  return text.Lines { lines = result }
end

function pretty.stat_parallel_prefix(cx, node)
  local result = terralib.newlist()
  result:insert(pretty.annotations(cx, node.annotations))
  result:insert(join({"__parallel_prefix(",
    commas({pretty.expr_region_root(cx, node.lhs),
            pretty.expr_region_root(cx, node.rhs),
            node.op, pretty.expr(cx, node.dir)}), ")"}))
  return text.Lines { lines = result }
end

function pretty.stat(cx, node)
  if node:is(ast.typed.stat.If) then
    return pretty.stat_if(cx, node)

  elseif node:is(ast.typed.stat.While) then
    return pretty.stat_while(cx, node)

  elseif node:is(ast.typed.stat.ForNum) then
    return pretty.stat_for_num(cx, node)

  elseif node:is(ast.typed.stat.ForNumVectorized) then
    return pretty.stat_for_num_vectorized(cx, node)

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

  elseif node:is(ast.typed.stat.Fence) then
    return pretty.stat_fence(cx, node)

  elseif node:is(ast.typed.stat.ParallelizeWith) then
    return pretty.stat_parallelize_with(cx, node)

  elseif node:is(ast.typed.stat.ParallelPrefix) then
    return pretty.stat_parallel_prefix(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function pretty.top_task_param(cx, node)
  cx:record_var(node, node.symbol)
  return cx:print_var(node.symbol) .. " : " .. tostring(node.param_type)
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
  for condition, values in node:items() do
    for _, value in values:items() do
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
      join({"replicable (", tostring(node.replicable), ")"}),
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
  local annotations = pretty.annotations(cx, node.annotations)

  local lines = terralib.newlist({annotations})
  lines:insert(join({((node.body and "") or "extern ") ..
                     "task " .. name, "(", params, ")", return_type }))
  if node.body then
    lines:insert(join({"-- ", commas(config_options) }))
  end
  if #meta > 0 then
    lines:insert(text.Line { value = "where" })
    lines:insert(text.Indent { value = commas(meta) })
    if node.body then
      lines:insert(text.Line { value = "do" })
    else
      lines:insert(text.Line { value = "end" })
    end
  end
  if node.body then
    lines:insert(pretty.block(cx, node.body))
    lines:insert(text.Line { value = "end" })
  end

  lines:insert(text.Line { value = "" }) -- Blank line

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
  local cx = pretty_context.new_global_scope()
  return render.entry(render_context.new_render_scope(), pretty.top(cx, node))
end

function pretty.entry_stat(node)
  local cx = pretty_context.new_global_scope()
  return render.entry(render_context.new_render_scope(), pretty.stat(cx, node))
end

function pretty.entry_expr(node)
  local cx = pretty_context.new_global_scope()
  return render.entry(render_context.new_render_scope(), pretty.expr(cx, node))
end

return pretty
