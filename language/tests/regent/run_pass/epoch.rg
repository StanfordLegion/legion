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

-- runs-with:
-- [["-ll:cpu", "4"]]

import "regent"

local c = regentlib.c

task g(s : region(int), z : int)
where reads(s), writes(s) do
  for y in s do
    @y += @y * z
  end
end

task k() : int
  var r = region(ispace(ptr, 1), int)
  var s = region(ispace(ptr, 3), int)

  var p = partition(equal, s, ispace(int1d, 3))

  r[0] = 1
  s[0] = 200
  s[1] = 30000
  s[2] = 4000000

  must_epoch
    __demand(__index_launch)
    for i = 0, 3 do
      g(p[i], 20)
    end
    g(r, 10)
  end
  return r[0] + s[0] + s[1] + s[2]
end

task main()
  regentlib.assert(k() == 84634211, "test failed")
end
regentlib.start(main)
