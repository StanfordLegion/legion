-- Copyright 2023 Stanford University
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

-- fails-with:
-- type_mismatch_partition_by_image6.rg:34: type mismatch in argument 1: expected region with structured indexspace but got one with unstructured indexspace
--   var q = image(r, p, b)
--               ^

import "regent"

local c = regentlib.c

task b()
end

task f()
  var r = region(ispace(ptr, 5), int)
  var s = region(ispace(ptr, 5), ptr(int, r))
  var sc = c.legion_coloring_create()
  var p = partition(disjoint, s, sc)
  c.legion_coloring_destroy(sc)

  var q = image(r, p, b)
end
