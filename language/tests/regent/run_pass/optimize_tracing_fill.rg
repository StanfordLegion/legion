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

-- runs-with:
-- [
--  [ "-dm:memoize", "-ll:cpu", "2" ],
--  [ "-dm:memoize", "-ll:cpu", "2", "-lg:no_fence_elision" ],
--  [ "-dm:memoize", "-ll:cpu", "2", "-lg:no_trace_optimization" ]
-- ]

import "regent"
import "bishop"

mapper
end

fspace fs
{
  input : int;
  output : int;
}

task init(r : region(ispace(int1d), fs))
where reads writes(r)
do
  for e in r do
    e.input = 0
    e.output = 0
  end
end

task inc(r : region(ispace(int1d), fs))
where reads(r.input), writes(r.output)
do
  for e in r do
    e.output = e.input + 1
  end
end

task step(r : region(ispace(int1d), fs))
where writes(r.input), reads(r.output)
do
  for e in r do
    e.input = e.output
  end
end

task check(r : region(ispace(int1d), fs), n : int)
where reads writes(r)
do
  for e in r do
    regentlib.assert(e.input % 3 == n, "test, failed")
  end
end

task main()
  var n = 2
  var r = region(ispace(int1d, n), fs)
  var cs = ispace(int1d, n)
  var p = partition(equal, r, cs)
  var q = partition(equal, r, cs)

  for k = 0, 10 do
    __demand(__trace)
    for i = 0, 3 do
      fill(r.input, 0)
      fill(r.output, 0)
      for color in cs do inc(q[color]) end
      for color in cs do step(q[color]) end
      for color in cs do inc(q[color]) end
      for color in cs do step(q[color]) end
      for color in cs do inc(q[color]) end
      for color in cs do step(q[color]) end
    end
    for color in cs do check(q[color], 0) end
  end
end
regentlib.start(main, bishoplib.make_entry())
