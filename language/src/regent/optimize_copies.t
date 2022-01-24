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

-- Regent Copy Optimizer
--
-- Attempts to merge copies on individual fields into a single copy
-- for multiple fields

local ast = require("regent/ast")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")

local optimize_copies = {}

local function is_copy(node)
  return node:is(ast.typed.stat.Expr) and
         node.expr:is(ast.typed.expr.Copy)
end

local function is_admissible_copy(copy)
  assert(copy:is(ast.typed.expr.Copy))
  return
    copy.src.region:is(ast.typed.expr.ID) and #copy.src.fields == 1 and
    copy.dst.region:is(ast.typed.expr.ID) and #copy.dst.fields == 1 and
    data.all(unpack(copy.conditions:map(function(condition)
      return condition.value:is(ast.typed.expr.ID)
    end)))
end

local function merge_copies(copies)
  assert(#copies >= 1)
  if #copies == 1 then return copies[1] end

  local first_copy = copies[1].expr
  local src = first_copy.src
  local src_fields = terralib.newlist()
  src_fields:insertall(src.fields)
  local dst = first_copy.dst
  local dst_fields = terralib.newlist()
  dst_fields:insertall(dst.fields)
  local op = first_copy.op
  local conditions = terralib.newlist()
  conditions:insertall(first_copy.conditions)

  for idx = 2, #copies do
    local copy = copies[idx].expr
    assert(src.region.value == copy.src.region.value)
    src_fields:insertall(copy.src.fields)
    assert(dst.region.value == copy.dst.region.value)
    dst_fields:insertall(copy.dst.fields)
    assert(op == copy.op)
    conditions:insertall(copy.conditions)
  end

  local new_copy = copies[1] {
    expr = first_copy {
      src = src {
        fields = src_fields,
      },
      dst = dst {
        fields = dst_fields,
      },
      op = op,
      conditions = conditions,
    }
  }
  return new_copy
end

function optimize_copies.block(node)
  local stats = node.stats
  local groups = {}

  for idx = 1, #stats do
    if is_copy(stats[idx]) and is_admissible_copy(stats[idx].expr) then
      groups[idx] = terralib.newlist {idx}
    end
  end

  local prev_idx = -1
  local prev_condition_variables = {}
  local prev_src = false
  local prev_dst = false
  local prev_op = false

  for idx = #stats, 1, -1 do
    local node = stats[idx]
    if is_copy(node) and is_admissible_copy(node.expr) then
      local copy = node.expr
      -- the previous copy group did not get killed
      if prev_idx ~= -1 and
         prev_src == copy.src.region.value and
         prev_dst == copy.dst.region.value and
         prev_op == copy.op then
        groups[prev_idx]:insertall(groups[idx])
        groups[idx] = groups[prev_idx]
        groups[prev_idx] = false
      end

      prev_idx = idx
      for cidx = 1, #copy.conditions do
        assert(copy.conditions[cidx].value:is(ast.typed.expr.ID))
        prev_condition_variables[copy.conditions[cidx].value.value] = true
      end
      assert(copy.src.region:is(ast.typed.expr.ID))
      prev_src = copy.src.region.value
      assert(copy.dst.region:is(ast.typed.expr.ID))
      prev_dst = copy.dst.region.value
      prev_op = copy.op

    -- check if the assignment defines any values that the copy group might use.
    -- if so, we cannot move the group above this assignment
    elseif node:is(ast.typed.stat.Assignment) then
      if not node.lhs:is(ast.typed.expr.ID) or
         prev_condition_variables[node.lhs.value] then
        prev_idx = -1
        prev_condition_variables = {}
        prev_src = false
        prev_dst = false
        prev_op = false
      end

    -- havoc the analysis state for any statements other than copy or assignment
    else
      prev_idx = -1
      prev_condition_variables = {}
      prev_src = false
      prev_dst = false
        prev_op = false
    end
  end

  local new_stats = terralib.newlist()
  for idx = 1, #stats do
    if not is_copy(stats[idx]) then
      new_stats:insert(stats[idx])
    elseif groups[idx] then
      new_stats:insert(merge_copies(groups[idx]:map(function(cidx) return stats[cidx] end)))
    end
  end

  return node { stats = new_stats }
end

function optimize_copies.top_task(node)
  local body = node.body
  body = ast.map_node_postorder(function(node)
    if node:is(ast.typed.Block) then
      if #data.filter(is_copy, node.stats) > 1 then
        return optimize_copies.block(node)
      else
        return node
      end
    else
      return node
    end
  end, node.body)
  return node { body = body }
end

function optimize_copies.top(node)
  if node:is(ast.typed.top.Task) then
    return optimize_copies.top_task(node)
  else
    return node
  end
end

function optimize_copies.entry(node)
  return optimize_copies.top(node)
end

optimize_copies.pass_name = "optimize_copies"

return optimize_copies
