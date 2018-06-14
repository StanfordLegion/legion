-- Copyright 2018 Stanford University
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

local alpha_convert = require("regent/alpha_convert")
local ast = require("regent/ast")
local check_annotations = require("regent/check_annotations")
local codegen = require("regent/codegen")
local parser = require("regent/parser")
local passes_hooks = require("regent/passes_hooks")
local pretty = require("regent/pretty")
local specialize = require("regent/specialize")
local normalize = require("regent/normalize")
local inline_tasks = require("regent/inline_tasks")
local eliminate_dead_code = require("regent/eliminate_dead_code")
local std = require("regent/std")
local type_check = require("regent/type_check")
local validate = require("regent/validate")
local profile = require("regent/profile")

local passes = {}

function passes.optimize(node)
  return passes_hooks.run_optimizations(node)
end

function passes.codegen(node, allow_pretty)
  if allow_pretty == nil then allow_pretty = true end

  if allow_pretty and std.config["pretty"] then print(profile("pretty", node, pretty.entry)(node)) end
  if std.config["validate"] then profile("validate", node, validate.entry)(node) end
  return profile("codegen", node, codegen.entry)(node)
end

function passes.compile(node, allow_pretty)
  local function ctor(environment_function)
    local env = environment_function()
    local node = profile("specialize", node, specialize.entry)(env, node)
    node = profile("normalize_after_specialize", node, normalize.entry)(node)
    if std.config["inline"] then
      node = profile("inline_tasks", node, inline_tasks.entry)(node)
    end
    profile("alpha_convert", node, alpha_convert.entry)(node) -- Run this here to avoid bitrot (discard result).
    node = profile("type_check", node, type_check.entry)(node)
    if std.config["inline"] then
      node = profile("eliminate_dead_code", node, eliminate_dead_code.entry)(node)
    end
    node = profile("normalize_after_type_check", node, normalize.entry)(node)
    node = profile("check_annotations", node, check_annotations.entry)(node)
    node = passes.optimize(node)
    return passes.codegen(node, allow_pretty)
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
