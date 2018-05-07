-- Copyright 2018 Stanford University, NVIDIA Corporation
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

local profile = require("regent/profile")

profile.set_import_time() -- Mark this as the first time we are entering into the Regent compiler

local builtins = require("regent/builtins")
local passes = require("regent/passes")
local passes_default = require("regent/passes_default")
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
    "__demand",
    "__forbid",
    "extern",
    "fspace",
    "rexpr",
    "rquote",
    "task",
  },
  keywords = {
    "__block",
    "__context",
    "__cuda",
    "__delete",
    "__demand",
    "__execution",
    "__external",
    "__fence",
    "__fields",
    "__forbid",
    "__inline",
    "__inner",
    "__leaf",
    "__mapping",
    "__openmp",
    "__optimize",
    "__parallel",
    "__parallelize_with",
    "__physical",
    "__raw",
    "__runtime",
    "__spmd",
    "__trace",
    "__unroll",
    "__vectorize",
    "acquire",
    "aliased",
    "allocate_scratch_fields",
    "adjust",
    "advance",
    "arrive",
    "await",
    "arrives",
    "attach",
    "awaits",
    "atomic",
    "copy",
    "cross_product",
    "cross_product_array",
    "detach",
    "disjoint",
    "dynamic_cast",
    "dynamic_collective",
    "dynamic_collective_get_result",
    "exclusive",
    "extern",
    "equal",
    "fill",
    "hdf5",
    "image",
    "index_type", -- reserved for future use
    "isnull",
    "ispace",
    "list_cross_product",
    "list_cross_product_complete",
    "list_slice_partition",
    "list_duplicate_partition",
    "list_invert",
    "list_phase_barriers",
    "list_range",
    "list_ispace",
    "list_from_element",
    "max",
    "min",
    "must_epoch",
    "new",
    "no_access_flag",
    "null",
    "partition",
    "phase_barrier",
    "preimage",
    "product",
    "ptr", -- reserved for future use
    "reads",
    "reduces",
    "relaxed",
    "release",
    "region",
    "simultaneous",
    "static_cast",
    "unsafe_cast",
    "wild", -- reserved for future use
    "with_scratch_fields",
    "where",
    "writes",
  },
}

function language:expression(lex)
  return passes.entry_expr(lex)
end

function language:statement(lex)
  return passes.entry_stat(lex)
end

function language:localstatement(lex)
  return passes.entry_stat(lex)
end

return language
