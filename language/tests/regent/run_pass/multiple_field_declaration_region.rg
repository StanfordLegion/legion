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

fspace foo(r : region(int)) {
  -- Note: The type is multiple-evaluated to produce distinct regions.
  {s, t} : region(int),
}
where s <= r, t <= r, s * t end

task sum(d : region(int), e : foo(d))
where reads writes(d) do
  var {f = s, g = t} = e
  var h = 0
  for i in f do h += @i end
  for j in g do h += @j end
  return h
end

task main()
  var a = region(ispace(ptr, 5), int)
  var p = partition(equal, a, ispace(int1d, 2))
  var b = p[0]
  var c = p[1]

  fill(a, 1)
  fill(b, 10)
  fill(c, 100)

  var x = [foo(a)] { s = b, t = c }

  var t = sum(a, x)
  regentlib.c.printf("result: %d\n", t)
end
regentlib.start(main)
