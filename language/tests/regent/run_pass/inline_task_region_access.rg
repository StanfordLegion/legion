-- Copyright 2024 Stanford University
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

__demand(__inline)
task inlined(caller_context : uint64) : int1d
  return [int1d]([int64](c.legion_context_get_unique_id(__context()) == caller_context))
end

task run(r : region(ispace(int1d), int))
where
  reads writes(r)
do
  var my_context : uint64 = c.legion_context_get_unique_id(__context())
  r[inlined(my_context)] = 1
  r[__forbid(__inline, inlined(my_context))] = 2
end

task check(r : region(ispace(int1d), int))
where
  reads(r)
do
  regentlib.assert(r[0] == 2, "test failed")
  regentlib.assert(r[1] == 1, "test failed")
  regentlib.assert(r[2] == 0, "test failed")
end

task main()
  var r = region(ispace(int1d, 3), int)
  fill(r, 0)
  run(r)
  check(r)
end

regentlib.start(main)
