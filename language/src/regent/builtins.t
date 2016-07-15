-- Copyright 2016 Stanford University
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
builtins.cross_product = std.cross_product
builtins.disjoint = std.disjoint
builtins.dynamic_collective = std.dynamic_collective
builtins.index_type = std.index_type
builtins.int1d = std.int1d
builtins.int2d = std.int2d
builtins.int3d = std.int3d
builtins.rect1d = std.rect1d
builtins.rect2d = std.rect2d
builtins.rect3d = std.rect3d
builtins.ispace = std.ispace
builtins.partition = std.partition
builtins.phase_barrier = std.phase_barrier
builtins.ptr = std.ptr
builtins.region = std.region
builtins.wild = std.wild

return builtins
