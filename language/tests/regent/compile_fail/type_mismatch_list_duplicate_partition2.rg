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
-- type_mismatch_list_duplicate_partition2.rg:28: type mismatch: expected list(int32) but got bool
--   var x = list_duplicate_partition(p, true)
--                                  ^

import "regent"

local c = regentlib.c

task f()
  var r = region(ispace(ptr, 5), int)
  var rc = c.legion_coloring_create()
  var p = partition(disjoint, r, rc)
  var x = list_duplicate_partition(p, true)
end
