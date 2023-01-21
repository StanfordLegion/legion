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

fspace Fields
{
  A : double,
  B : double,
  C : double,
  D : &double,
}

__demand(__inline)
task test(t : region(ispace(int1d), Fields))
where
  reads(t.B), writes(t.A)
do
  var a = __fields(t.{A, B})
  return a[0]
end

__demand(__inline)
task test2(t : region(ispace(int1d), Fields))
where
  reads(t.D)
do
  var a = __fields(t.{D})
  return a[0]
end

task main()
  var t = region(ispace(int1d, 2), Fields)
  fill(t.{A, B, C}, 0)
  var f = test(t)
  var g = test2(t)
  regentlib.assert(f == __fields(t.A)[0], "test failed")
  regentlib.assert(g == __fields(t.D)[0], "test failed")
end

regentlib.start(main)
