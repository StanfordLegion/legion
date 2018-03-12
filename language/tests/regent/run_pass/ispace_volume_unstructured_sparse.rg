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

-- Test volume of index spaces that are not dense.

import "regent"

task main()
  var r = region(ispace(ptr, 5), int1d)
  r[0] = 0
  r[1] = 0
  r[2] = 1
  r[3] = 0
  r[4] = 0
  var p = partition(r, ispace(int1d, 2))
  var s = p[0]
  regentlib.c.printf("r.ispace.volume: %d\n", r.ispace.volume)
  regentlib.assert(r.ispace.volume == 5, "test failed")

  regentlib.c.printf("s.ispace.volume: %d\n", s.ispace.volume)
  regentlib.assert(s.ispace.volume == 4, "test failed")
end
regentlib.start(main)
