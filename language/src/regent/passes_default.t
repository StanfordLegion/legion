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

-- Regent Compiler Passes (Default List of Passes)

local passes_hooks = require("regent/passes_hooks")
local std = require("regent/std")

local check_parallelizable = require("regent/check_parallelizable")
local copy_propagate = require("regent/copy_propagate")
local optimize_config_options = require("regent/optimize_config_options")
local optimize_divergence = require("regent/optimize_divergence")
local optimize_futures = require("regent/optimize_futures")
local optimize_predicate = require("regent/optimize_predicate")
local optimize_index_launches = require("regent/optimize_index_launches")
local optimize_mapping = require("regent/optimize_mapping")
local optimize_traces = require("regent/optimize_traces")
local parallelize_tasks = require("regent/parallelize_tasks")
local skip_empty_tasks = require("regent/skip_empty_tasks")
local vectorize_loops = require("regent/vectorize_loops")

if std.config["flow"] then
  require("regent/flow_from_ast") -- priority 15
  require("regent/flow_spmd")     -- priority 16
  require("regent/flow_to_ast")   -- priority 24
end

local needs_check_parallelizable =
  std.config["parallelize"] or std.config["openmp"] or std.config["cuda"] or std.config["vectorize"]

local needs_optimize_config_options =
  needs_check_parallelizable or std.config["leaf"] or std.config["inner"] or std.config["replicable"]

if needs_optimize_config_options then passes_hooks.add_optimization(8, optimize_config_options) end
if needs_check_parallelizable then
  passes_hooks.add_optimization(9, check_parallelizable)
  passes_hooks.add_optimization(10, copy_propagate)
end
if std.config["parallelize"] then passes_hooks.add_optimization(14, parallelize_tasks) end
if std.config["index-launch"] then passes_hooks.add_optimization(25, optimize_index_launches) end
if std.config["skip-empty-tasks"] then passes_hooks.add_optimization(28, skip_empty_tasks) end
if std.config["future"] then passes_hooks.add_optimization(30, optimize_futures) end
if std.config["predicate"] then passes_hooks.add_optimization(40, optimize_predicate) end
if std.config["mapping"] then passes_hooks.add_optimization(50, optimize_mapping) end
if std.config["trace"] then passes_hooks.add_optimization(60, optimize_traces) end
if std.config["no-dynamic-branches"] then passes_hooks.add_optimization(70, optimize_divergence) end
if needs_check_parallelizable and std.config["flow"] then passes_hooks.add_optimization(75, check_parallelizable) end
if std.config["vectorize"] then passes_hooks.add_optimization(80, vectorize_loops) end

passes_hooks.debug_optimizations()
