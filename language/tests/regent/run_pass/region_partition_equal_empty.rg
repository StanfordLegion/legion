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

task main()
  var r = region(ispace(int1d, 0), int)
  var cs = ispace(int1d, 2)
  var p = partition(equal, r, cs)

  var s = 0
  for i in cs do
    var t = 0
    for x in p[i] do
      t += 1
    end
    regentlib.assert(t == 0, "test failed")
    s += 1
  end
  regentlib.assert(s == 2, "test failed")
end
regentlib.start(main)
