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
-- [["-ll:gpu", "1" ]]

import "regent"

__demand(__cuda)
task init(r : region(ispace(int2d), double),
          v : double)
where
  reads writes(r)
do
  for e in r do
    @e = v
  end
end

__demand(__cuda)
task red(r : region(ispace(int2d), double))
where
  reads(r)
do
  var sum : double = 0.0
  for e in r do
    sum += @e
  end
  return sum
end

task main()
  var size = 10
  var v : double = 2.0
  var r = region(ispace(int2d, {size, size}), double)
  init(r, v)
  var res = red(r)
  regentlib.assert(res == r.volume * v, "test failed")
end

regentlib.start(main)
