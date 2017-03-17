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

-- Regent Compiler Passes (Default List of Passes)

local passes_hooks = require("regent/passes_hooks")
local std = require("regent/std")

local inline_tasks = require("regent/inline_tasks")
local optimize_config_options = require("regent/optimize_config_options")
local optimize_divergence = require("regent/optimize_divergence")
local optimize_futures = require("regent/optimize_futures")
local optimize_index_launches = require("regent/optimize_index_launches")
local optimize_mapping = require("regent/optimize_mapping")
local optimize_traces = require("regent/optimize_traces")
local parallelize_tasks = require("regent/parallelize_tasks")
local vectorize_loops = require("regent/vectorize_loops")

if std.config["flow"] then
  require("regent/flow_from_ast") -- priority 15
  require("regent/flow_spmd")     -- priority 16
  require("regent/flow_to_ast")   -- priority 24
end

if std.config["inline"] then passes_hooks.add_optimization(1, inline_tasks) end
if std.config["parallelize"] then passes_hooks.add_optimization(10, parallelize_tasks) end
if std.config["index-launch"] then passes_hooks.add_optimization(25, optimize_index_launches) end
if std.config["future"] then passes_hooks.add_optimization(30, optimize_futures) end
if std.config["leaf"] or std.config["inner"] then passes_hooks.add_optimization(40, optimize_config_options) end
if std.config["mapping"] then passes_hooks.add_optimization(50, optimize_mapping) end
if std.config["trace"] then passes_hooks.add_optimization(60, optimize_traces) end
if std.config["no-dynamic-branches"] then passes_hooks.add_optimization(70, optimize_divergence) end
if std.config["vectorize"] then passes_hooks.add_optimization(80, vectorize_loops) end

passes_hooks.debug_optimizations()
