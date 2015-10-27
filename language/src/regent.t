-- Copyright 2015 Stanford University, NVIDIA Corporation
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

-- Regent Language Entry Point

local builtins = require("regent/builtins")
local passes = require("regent/passes")
local std = require("regent/std")

-- Add Language Builtins to Global Environment

local function add_builtin(k, v)
  assert(rawget(_G, k) == nil, "Builtin " .. tostring(k) .. " already defined")
  rawset(_G, k, v)
end

for k, v in pairs(builtins) do
  add_builtin(k, v)
end
add_builtin("regentlib", std)

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
    "__cuda",
    "__demand",
    "__fields",
    "__forbid",
    "__inline",
    "__parallel",
    "__physical",
    "__raw",
    "__runtime",
    "__spmd",
    "__unroll",
    "__vectorize",
    "aliased", -- reserved for future use
    "advance",
    "arrives",
    "awaits",
    "atomic",
    "cross_product",
    "disjoint", -- reserved for future use
    "dynamic_cast",
    "exclusive",
    "index_type", -- reserved for future use
    "isnull",
    "ispace",
    "max",
    "min",
    "must_epoch",
    "new",
    "null",
    "partition",
    "phase_barrier",
    "product",
    "ptr", -- reserved for future use
    "reads",
    "reduces",
    "relaxed",
    "region",
    "simultaneous",
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
  return passes.compile(lex)
end

function language:localstatement(lex)
  return passes.compile(lex)
end

return language
