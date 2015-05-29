-- Copyright 2015 Stanford University
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

-- Legion Language Entry Point

local ast = require("legion/ast")
local builtins = require("legion/builtins")
local codegen = require("legion/codegen")
local optimize_config_options = require("legion/optimize_config_options")
local optimize_divergence = require("legion/optimize_divergence")
local optimize_futures = require("legion/optimize_futures")
local optimize_inlines = require("legion/optimize_inlines")
local optimize_loops = require("legion/optimize_loops")
local parser = require("legion/parser")
local specialize = require("legion/specialize")
local std = require("legion/std")
local type_check = require("legion/type_check")
local vectorize_loops = require("legion/vectorize_loops")
local inline_tasks = require("legion/inline_tasks")

-- Add Language Builtins to Global Environment

for k, v in pairs(builtins) do
  assert(rawget(_G, k) == nil, "Builtin " .. tostring(k) .. " already defined")
  _G[k] = v
end

-- Add Interface to Helper Functions

_G["legionlib"] = std

-- Compiler

function compile(lex)
  local node = parser:parse(lex)
  local function ctor(environment_function)
    local env = environment_function()
    local ast = specialize.entry(env, node)
    ast = type_check.entry(ast)
    if std.config["index-launches"] then ast = optimize_loops.entry(ast) end
    if std.config["futures"] then ast = optimize_futures.entry(ast) end
    if std.config["inlines"] then ast = optimize_inlines.entry(ast) end
    if std.config["leaf"] then ast = optimize_config_options.entry(ast) end
    if std.config["no-dynamic-branches"] then ast = optimize_divergence.entry(ast) end
    if std.config["task-inlines"] then ast = inline_tasks.entry(ast) end
    if std.config["vectorize"] then ast = vectorize_loops.entry(ast) end
    ast = codegen.entry(ast)
    return ast
  end
  return ctor, {node.name}
end

-- Language Definition

-- Note: Keywords marked "reserved for future use" are usually marked
-- as such to indicate that those words are exposed as builtin types,
-- and reserved in case types are ever inducted into the main
-- language.
local language = {
  name = "legion",
  entrypoints = {
    "task",
    "fspace",
    "__demand",
  },
  keywords = {
    "__context",
    "__demand",
    "__fields",
    "__parallel",
    "__vectorize",
    "__physical",
    "__runtime",
    "__inline",
    "__cuda",
    "aliased", -- reserved for future use
    "cross_product",
    "disjoint", -- reserved for future use
    "dynamic_cast",
    "isnull",
    "ispace",
    "max",
    "min",
    "new",
    "null",
    "partition",
    "ptr", -- reserved for future use
    "reads",
    "reduces",
    "region",
    "static_cast",
    "wild", -- reserved for future use
    "where",
    "writes",
  },
}

-- function language:expression(lex)
--   return function(environment_function)
--   end
-- end

function language:statement(lex)
  return compile(lex)
end

function language:localstatement(lex)
  return compile(lex)
end

return language
