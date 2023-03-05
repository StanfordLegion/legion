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
-- invalid_import_partition1.rg:30: $raw_p is not a disjoint partition

import "regent"

task main()
  var is = ispace(int1d, 5)
  var r = region(is, int)
  var t : transform(1, 1)
  t[{0, 0}] = 0
  var e = r.bounds
  var cs = ispace(int1d, 2)
  var p = restrict(r, t, e, cs)

  var raw_p = __raw(p)
  var q = __import_partition(disjoint, r, cs, raw_p)
end

regentlib.start(main)
