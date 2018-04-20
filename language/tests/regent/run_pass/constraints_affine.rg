-- Copyright 2018 Stanford University
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

fspace assert_subregion {
  x : region(int),
  y : region(int),
} where x <= y end

fspace assert_disjoint {
  x : region(int),
  y : region(int),
} where x * y end

task main()
  var t = region(ispace(ptr, 5), int)
  var p = partition(equal, t, ispace(int1d, 5))

  assert_disjoint { x = p[0], y = p[1] }

  for i = 1, 2 do
    assert_disjoint { x = p[i], y = p[i + 1] }
    assert_disjoint { x = p[i - 1], y = p[i + 0] }
    -- assert_disjoint { x = p[i], y = p[i - 1] } -- FIXME: name already defined?
  end
end
regentlib.start(main)
