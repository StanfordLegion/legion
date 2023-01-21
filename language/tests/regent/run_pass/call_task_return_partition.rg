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

task make_partition(r : region(int), s : ispace(int1d))
  return partition(equal, r, s)
end

task main()
  var c = ispace(int1d, 3)
  var a = region(ispace(ptr, 5), int)
  var b = make_partition(a, c)
  fill(a, 17)
  var t = 0
  for i in c do
    var d = b[i]
    for x in d do
      t += @x
    end
  end
  regentlib.assert(t == 5*17, "test failed")
end
regentlib.start(main)
