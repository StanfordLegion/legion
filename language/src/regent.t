-- Copyright 2022 Stanford University, NVIDIA Corporation
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
    "__allow",
    "__forbid",
    "__line",
    "extern",
    "fspace",
    "rexpr",
    "remit",
    "rquote",
    "task",
  },
  keywords = {
    "__block",
    "__constant_time_launch",
    "__context",
    "__cuda",
    "__delete",
    "__demand",
    "__execution",
    "__fence",
    "__fields",
    "__forbid",
    "__future",
    "__idempotent",
    "__import_ispace",
    "__import_region",
    "__import_partition",
    "__import_cross_product",
    "__index_launch",
    "__inline",
    "__inner",
    "__leaf",
    "__local",
    "__line",
    "__mapping",
    "__openmp",
    "__optimize",
    "__parallel",
    "__parallelize_with",
    "__parallel_prefix",
    "__physical",
    "__predicate",
    "__raw",
    "__replicable",
    "__runtime",
    "__spmd",
    "__task",
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
    "complete",
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
    "incomplete",
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
    "reads",
    "reduces",
    "relaxed",
    "release",
    "rescape",
    "region",
    "restrict",
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
  return passes.entry_stat(lex, false)
end

function language:localstatement(lex)
  return passes.entry_stat(lex, true)
end

return language
