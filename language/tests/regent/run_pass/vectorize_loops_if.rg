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

-- FIXME: Turn off this test until vectorizer supports scattered reads again.
-- runs-with:
-- []
--

import "regent"

struct vec2
{
  x : float,
  y : float,
}

fspace fs2
{
  v : vec2,
}

fspace fs1(r : region(fs2))
{
  v1 : vec2,
  v2 : vec2,
  p : ptr(fs2, r),
}

task f(r2 : region(fs2), r : region(fs1(r2)))
where
  reads(r, r2),
  writes(r, r2)
do
  var x : float = 10.0
  var opt1 : bool = true
  var opt2 : bool = false
  var opt3 : bool = true

  __demand(__vectorize)
  for e in r do
    var x = e.v1
    var p = e.p
    var y = p.v
    var z : vec2
    if opt1 then
      z.x = x.x + y.x
      z.y = x.y + y.y
    elseif opt2 then
      z.y = x.x + y.x
      z.x = x.y + y.y
    end

    if opt3 then
      e.v1 = z
    end
  end

  __forbid(__vectorize)
  for e in r do
    var x = e.v2
    var p = e.p
    var y = p.v
    var z : vec2
    if opt1 then
      z.x = x.x + y.x
      z.y = x.y + y.y
    elseif opt2 then
      z.y = x.x + y.x
      z.x = x.y + y.y
    end

    if opt3 then
      e.v2 = z
    end
  end
end

task toplevel()
  var n = 10
  var r2 = region(ispace(ptr, n), fs2)
  var r1 = region(ispace(ptr, n), fs1(r2))
  for i = 0, n do
    var ptr2 = dynamic_cast(ptr(fs2, r2), i)
    ptr2.v.x = 1.0 + i
    ptr2.v.y = 2.0 + i

    var ptr1 = dynamic_cast(ptr(fs1(r2), r1), i)
    ptr1.v1.x = 1.0
    ptr1.v1.y = 2.0
    ptr1.v2.x = 1.0
    ptr1.v2.y = 2.0
    ptr1.p = ptr2
  end
  f(r2, r1)
  for e in r1 do
    regentlib.assert(e.v1.x == e.v2.x, "test failed")
    regentlib.assert(e.v1.y == e.v2.y, "test failed")
  end
end

regentlib.start(toplevel)
