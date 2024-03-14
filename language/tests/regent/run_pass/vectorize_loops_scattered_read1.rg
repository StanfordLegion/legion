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

local c = regentlib.c

fspace fs1
{
  f : double,
}

fspace fs2(r : region(fs1))
{
  p : ptr(fs1, r),
  g : double,
  h : double,
}

task init(r1 : region(fs1), r2 : region(fs2(r1)), size : int)
where reads writes(r1, r2)
do
  for e in r1 do e.f = c.drand48() end
  for e in r2 do
    e.p = unsafe_cast(ptr(fs1, r1), ptr(__raw(e).value))
    e.g = 0
    e.h = 0
  end
end

task stencil(r1 : region(fs1), r2 : region(fs2(r1)))
where reads(r1.f, r2.p), writes(r2.{g, h})
do
  __demand(__vectorize)
  for e in r2 do
    e.g = 0.5 * r1[e.p].f
  end

  __forbid(__vectorize)
  for e in r2 do
    e.h = 0.5 * r1[e.p].f
  end
end

task check(r1 : region(fs1), r2 : region(fs2(r1)))
where reads(r2.{g, h})
do
  for e in r2 do
    regentlib.assert(e.g == e.h, "test failed")
  end
end

task test(size : int)
  var r1 = region(ispace(ptr, size), fs1)
  var r2 = region(ispace(ptr, size), fs2(wild))
  init(r1, r2, size)
  stencil(r1, r2)
  check(r1, r2)
end

task toplevel()
  test(100)
end

regentlib.start(toplevel)
