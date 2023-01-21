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

import "regent"

local c = regentlib.c

task test_partition()
  var r = region(ispace(ptr, 4), int)
  var p = partition(equal, r, ispace(int1d, 3))

  var qc = c.legion_coloring_create()
  c.legion_coloring_ensure_color(qc, 0)
  var q = partition(disjoint, p[0], qc)
  c.legion_coloring_destroy(qc)
end

task test_partition_by_field()
  var r = region(ispace(ptr, 4), int1d)
  fill(r, 0)
  var p = partition(equal, r, ispace(int1d, 3))

  var q = partition((p[0]), ispace(int1d, 2))
end

task main()
  test_partition()
  test_partition_by_field()
end
regentlib.start(main)
