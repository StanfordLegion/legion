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

local codegen = require("regent/codegen")
local flow_dead_code_elimination = require("regent/flow_dead_code_elimination")
local flow_from_ast = require("regent/flow_from_ast")
local flow_loop_fusion = require("regent/flow_loop_fusion")
local flow_spmd = require("regent/flow_spmd")
local flow_task_fusion = require("regent/flow_task_fusion")
local flow_to_ast = require("regent/flow_to_ast")
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

function passes.optimize(ast)
  if std.config["task-inlines"] then ast = inline_tasks.entry(ast) end
  if std.config["flow"] then
    ast = flow_from_ast.entry(ast)
    if std.config["flow-spmd"] then ast = flow_spmd.entry(ast) end
    -- ast = flow_loop_fusion.entry(ast)
    -- ast = flow_task_fusion.entry(ast)
    -- ast = flow_dead_code_elimination.entry(ast)
    ast = flow_to_ast.entry(ast)
  end
  if std.config["index-launches"] then ast = optimize_loops.entry(ast) end
  if std.config["futures"] then ast = optimize_futures.entry(ast) end
  if std.config["leaf"] then ast = optimize_config_options.entry(ast) end
  if std.config["inlines"] then ast = optimize_inlines.entry(ast) end
  if std.config["trace"] then ast = optimize_traces.entry(ast) end
  if std.config["no-dynamic-branches"] then ast = optimize_divergence.entry(ast) end
  if std.config["vectorize"] then ast = vectorize_loops.entry(ast) end
  return ast
end

function passes.compile(lex)
  local node = parser:parse(lex)
  local function ctor(environment_function)
    local env = environment_function()
    local ast = specialize.entry(env, node)
    ast = type_check.entry(ast)
    ast = passes.optimize(ast)
    if std.config["pretty"] then print(pretty.entry(ast)) end
    return codegen.entry(ast)
  end
  return ctor, {node.name}
end

return passes
