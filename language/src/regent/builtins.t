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

-- Legion Builtins

local std = require("regent/std")

local builtins = {}

-- Builtins consists of a list of which will be stuffed into the
-- global scope of any legion program (i.e. they need not be accessed
-- via std).

builtins.aliased = std.aliased
builtins.complete = std.complete
builtins.complex = std.complex
builtins.complex32 = std.complex32
builtins.complex64 = std.complex64
builtins.cross_product = std.cross_product
builtins.disjoint = std.disjoint
builtins.dynamic_collective = std.dynamic_collective
builtins.incomplete = std.incomplete
builtins.index_type = std.index_type
for dim = 1, std.max_dim do
  builtins["int" .. dim .. "d"] = std["int" .. dim .. "d"]
  builtins["rect" .. dim .. "d"] = std["rect" .. dim .. "d"]
end
builtins.ispace = std.ispace
builtins.partition = std.partition
builtins.phase_barrier = std.phase_barrier
builtins.ptr = std.ptr
builtins.region = std.region
builtins.transform = std.transform
builtins.wild = std.wild

return builtins
