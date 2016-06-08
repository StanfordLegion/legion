-- Copyright 2016 Stanford University
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

fspace fs {
  x : double,
  y : double,
  z : double
}

__demand(__inline)
task foo(x : region(ispace(int1d), int), k : int)
  var size = 5
  var rtmp = region(ispace(int1d, size), int)
  for i in rtmp.ispace do
    rtmp[i] = [int](i) + k
  end
  var sum = 0
  for e in rtmp do
    sum += @e
  end
  return sum
end

task main()
  var size : int1d = 4
  var is = ispace(int1d, size)
  var r = region(is, int)

  var x = 0
  x += foo(r, 10)
  x += foo(r, 20)
  regentlib.assert(x == 170, "test failed")
end

regentlib.start(main)
