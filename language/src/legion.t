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
local parser = require("legion/parser")
local specialize = require("legion/specialize")
local std = require("legion/std")
local type_check = require("legion/type_check")

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
    local specialized = specialize.entry(env, node)
    local typed = type_check.entry(specialized)
    local code = codegen.entry(typed)
    return code
  end
  return ctor, {node.name}
end

-- Language Definition

local language = {
  name = "legion",
  entrypoints = {
    "task",
    "fspace",
  },
  keywords = {
    "__context",
    "__fields",
    "__physical",
    "__runtime",
    "isnull",
    "new",
    "null",
    "partition",
    "reads",
    "reduces",
    "region",
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
