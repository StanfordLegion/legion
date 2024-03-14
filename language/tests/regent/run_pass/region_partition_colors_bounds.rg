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

-- Tests fix for https://github.com/StanfordLegion/legion/issues/762

task f(r : region(ispace(int3d), int), p : partition(disjoint, r, ispace(int3d)))
where reads(r) do
  var c = p.colors
  var cb = c.bounds
  return cb
end

task main()
  var r = region(ispace(int3d, { 2, 3, 5 }), int)
  var p = partition(equal, r, ispace(int3d, { 2, 3, 5 }))
  fill(r, 0)
  var x = f(r, p)
  regentlib.assert(x.hi.z - x.lo.z + 1 == 5, "test failed")
end
regentlib.start(main)
