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

-- Regent Compiler Passes (Hooks)

local log = require("common/log")
local profile = require("regent/profile")

local passes_hooks = {}

local log_passes = log.make_logger("passes")

passes_hooks.optimization_hooks = terralib.newlist()

local function compare_priority(a, b)
  return a[1] < b[1]
end

function passes_hooks.add_optimization(priority, pass)
  assert(priority > 0 and priority < 100, "priority must be between 0 and 100")
  assert(type(pass.entry) == "function", "pass requires entry function")
  assert(type(pass.pass_name) == "string", "pass requires name")

  passes_hooks.optimization_hooks:insert({priority, pass})
  passes_hooks.optimization_hooks:sort(compare_priority)
end

function passes_hooks.run_optimizations(node)
  for _, hook in ipairs(passes_hooks.optimization_hooks) do
    node = profile(hook[2].pass_name, node, hook[2].entry)(node)
  end
  return node
end

function passes_hooks.debug_optimizations(node)
  log_passes:info("Registered optimizations passes:")
  for _, hook in ipairs(passes_hooks.optimization_hooks) do
    log_passes:info("  " .. tostring(hook[1]) .. ". " .. tostring(hook[2].pass_name))
  end
  return node
end

return passes_hooks
