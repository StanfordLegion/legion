-- Copyright 2019 Stanford University, NVIDIA Corporation
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

-- Data-flow analysis framework

local ast = require("regent/ast")
local std = require("regent/std")

local dataflow = {}

local forward_metatable = {}
local forward = setmetatable({}, forward_metatable)
forward.__index = forward
dataflow.forward = forward

-- TODO: Maybe make meet be in place

-- Forward data-flow analysis. The following methods are needed for the
-- semilattice:
-- +, meet operation
-- ==, check for equality of states (for determining convergence)
-- :copy, duplicate state
--
-- Additionally, it needs the following four methods to implement the transfer
-- function.
--
-- :enter_block
-- :exit_block
-- :statement (does not need to handle control flow)
--
-- After constructing an instance, which should only be used inside a single
-- function, use :block to run the analysis on a block of code.

function forward_metatable:__call(exit_state)
  local out = {
    block_entries = {},
    exit_stack = { exit_state, len = 1 },
  }

  return setmetatable(out, forward)
end

function forward:exit_state()
  return self.exit_stack[1]
end

local function push(stack, state)
  stack.len = stack.len + 1
  stack[stack.len] = state
end

local function pop(stack)
  local top = stack[stack.len]
  stack[stack.len] = nil
  stack.len = stack.len - 1
  return top
end

function forward:block(block, entry, dont_exit)
  if not entry then
    return nil
  end

  if self.block_entries[block] == entry then
    -- No need to redo this execution path.
    return nil
  end
  self.block_entries[block] = entry

  entry:enter_block(block)
  for _, stat in ipairs(block.stats) do
    entry = self:stat(stat, entry)
    if not entry then
      return nil
    end
  end
  if not dont_exit then
    entry:exit_block(block)
  end

  return entry
end

function forward:stat_if(node, entry)
  local cond = self:unused_expr(node.cond, entry)
  local exit = self:block(node.then_block, dataflow.copy(cond))

  for _, else_if in ipairs(node.elseif_blocks) do
    cond = self:unused_expr(else_if.cond, cond)
    local block = self:block(else_if.block, dataflow.copy(cond))
    exit = dataflow.meet(exit, block)
  end

  local else_block = self:block(node.else_block, cond)
  exit = dataflow.meet(exit, else_block)
  return exit
end

function forward:stat_while(node, entry)
  local exit
  repeat
    local cond = self:unused_expr(node.cond, dataflow.copy(entry))
    exit = dataflow.meet_copy(exit, cond)
    push(self.exit_stack, exit)
    local block = self:block(node.block, cond)
    exit = pop(self.exit_stack)

    local updated
    entry, updated = dataflow.meet_updated(entry, block)
  until not updated

  return exit
end

function forward:stat_repeat(node, entry)
  local exit
  repeat
    push(self.exit_stack, exit)
    local block = self:block(node.block, dataflow.copy(entry), true)
    local cond = self:unused_expr(node.until_cond, block)
    if cond then
      cond:exit_block(node.block)
    end
    exit = dataflow.meet(pop(self.exit_stack), cond)

    local updated
    entry, updated = dataflow.meet_updated(entry, cond)
  until not updated

  return exit
end

function forward:stat_for_num(node, entry)
  for _, value in ipairs(node.values) do
    entry = self:unused_expr(value, entry)
  end

  local exit
  repeat
    exit = dataflow.meet(exit, entry)
    push(self.exit_stack, exit)
    local block = self:block(node.block, dataflow.copy(entry))
    exit = pop(self.exit_stack)

    local updated
    entry, updated = dataflow.meet_updated(entry, block)
  until not updated

  return exit
end

function forward:stat_for_list(node, entry)
  entry = self:unused_expr(node.value, entry)

  local exit
  repeat
    exit = dataflow.meet(exit, entry)
    push(self.exit_stack, exit)
    local block = self:block(node.block, dataflow.copy(entry))
    exit = pop(self.exit_stack)

    local updated
    entry, updated = dataflow.meet_updated(entry, block)
  until not updated

  return exit
end

function forward:stat_block(node, entry)
  return self:block(node.block, entry)
end

function forward:stat_break(node, entry)
  entry:statement(node)
  push(self.exit_stack, dataflow.meet_copy(pop(self.exit_stack), entry))

  -- nil represents unreachable code.
  return nil
end

function forward:stat_return(node, entry)
  entry:statement(node)
  self.exit_stack[1] = dataflow.meet_copy(self.exit_stack[1], entry)

  return nil
end

function forward:stat_other(node, entry)
  entry:statement(node)
  return entry
end

local function unreachable()
  assert(false, "unreachable")
end

local forward_stat_node = {
  [ast.typed.stat.If]                = forward.stat_if,
  [ast.typed.stat.While]             = forward.stat_while,
  [ast.typed.stat.ForNum]            = forward.stat_for_num,
  [ast.typed.stat.ForNumVectorized]  = forward.stat_for_num,
  [ast.typed.stat.IndexLaunchNum]    = forward.stat_for_num,
  [ast.typed.stat.ForList]           = forward.stat_for_list,
  [ast.typed.stat.ForListVectorized] = forward.stat_for_list,
  [ast.typed.stat.IndexLaunchList]   = forward.stat_for_list,
  [ast.typed.stat.Repeat]            = forward.stat_repeat,
  [ast.typed.stat.Block]             = forward.stat_block,
  [ast.typed.stat.MustEpoch]         = forward.stat_block,
  [ast.typed.stat.ParallelizeWith]   = forward.stat_block,
  [ast.typed.stat.Break]             = forward.stat_break,
  [ast.typed.stat.Return]            = forward.stat_return,

  [ast.typed.stat.Var]               = forward.stat_other,
  [ast.typed.stat.VarUnpack]         = forward.stat_other,
  [ast.typed.stat.Expr]              = forward.stat_other,
  [ast.typed.stat.Assignment]        = forward.stat_other,
  [ast.typed.stat.Reduce]            = forward.stat_other,
  [ast.typed.stat.RawDelete]         = forward.stat_other,
  [ast.typed.stat.Fence]             = forward.stat_other,
  [ast.typed.stat.ParallelPrefix]    = forward.stat_other,
  [ast.typed.stat.BeginTrace]        = forward.stat_other,
  [ast.typed.stat.EndTrace]          = forward.stat_other,
  [ast.typed.stat.MapRegions]        = forward.stat_other,
  [ast.typed.stat.UnmapRegions]      = forward.stat_other,

  [ast.typed.stat.Elseif]   = unreachable,
  [ast.typed.stat.Internal] = unreachable,
}

local forward_stat = ast.make_single_dispatch(forward_stat_node, {ast.typed.stat})
function forward:stat(...)
  return forward_stat(self)(...)
end

function forward:unused_expr(node, entry)
  if not entry then
    return nil
  end

  entry:statement(
    ast.typed.stat.Expr {
      expr = node,
      span = node.span,
      annotations = node.annotations,
    })
  return entry
end

function dataflow.meet(x, y)
  if not y then
    return x
  elseif not x then
    return y
  else
    return x + y
  end
end

function dataflow.meet_copy(x, y)
  if not y then
    return x
  elseif not x then
    return y:copy()
  else
    return x + y
  end
end

function dataflow.meet_updated(prev, x)
  if not prev then
    return x, prev ~= x
  elseif not x then
    return prev, false
  else
    local new = prev + x
    return new, prev ~= new
  end
end

function dataflow.copy(x)
  if not x then
    return nil
  end
  return x:copy()
end

return dataflow
