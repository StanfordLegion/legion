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

fspace foo {
  x : bool,
  {y, z, w} : int,
  {u, v,} : double[2], -- Note: Test trailing comma.
}

task main()
  var f : foo
  f.x = true
  f.y, f.z, f.w = 1, 20, 300
  f.u[0], f.u[1] = 3.14, 2.71
  f.v[0], f.v[1] = 0.5, 0.25

  regentlib.assert(f.x, "test failed")
  regentlib.assert(f.y + f.z + f.w == 321, "test failed")
end
regentlib.start(main)
