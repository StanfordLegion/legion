-- Copyright 2016 Stanford University
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

-- Regent Compiler Passes

local ast = require("regent/ast")
local codegen = require("regent/codegen")
local optimize_config_options = require("regent/optimize_config_options")
local optimize_divergence = require("regent/optimize_divergence")
local optimize_futures = require("regent/optimize_futures")
local optimize_inlines = require("regent/optimize_inlines")
local optimize_loops = require("regent/optimize_loops")
local optimize_traces = require("regent/optimize_traces")
local parser = require("regent/parser")
local pretty = require("regent/pretty")
local specialize = require("regent/specialize")
local std = require("regent/std")
local type_check = require("regent/type_check")
local vectorize_loops = require("regent/vectorize_loops")
local inline_tasks = require("regent/inline_tasks")

local passes = {}

function passes.optimize(node)
  if std.config["task-inlines"] then node = inline_tasks.entry(node) end
  if std.config["index-launches"] then node = optimize_loops.entry(node) end
  if std.config["futures"] then node = optimize_futures.entry(node) end
  if std.config["leaf"] then node = optimize_config_options.entry(node) end
  if std.config["inlines"] then node = optimize_inlines.entry(node) end
  if std.config["trace"] then node = optimize_traces.entry(node) end
  if std.config["no-dynamic-branches"] then node = optimize_divergence.entry(node) end
  if std.config["vectorize"] then node = vectorize_loops.entry(node) end
  return node
end

function passes.compile(node, allow_pretty)
  local function ctor(environment_function)
    local env = environment_function()
    local node = specialize.entry(env, node)
    node = type_check.entry(node)
    node = passes.optimize(node)
    if allow_pretty and std.config["pretty"] then print(pretty.entry(node)) end
    return codegen.entry(node)
  end
  return ctor
end

function passes.entry_expr(lex)
  local node = parser:entry_expr(lex)
  local ctor = passes.compile(node, false)
  return ctor
end

function passes.entry_stat(lex)
  local node = parser:entry_stat(lex)
  local ctor = passes.compile(node, true)
  return ctor, {node.name}
end

return passes
