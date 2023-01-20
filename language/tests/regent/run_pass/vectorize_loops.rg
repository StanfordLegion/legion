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

struct vec2
{
  x : float,
  y : float,
}

fspace fs2
{
  v1 : vec2,
  v2 : vec2,
}

fspace fs1(ra : region(fs2), rb : region(fs2))
{
  v1 : vec2,
  v2 : vec2,
  p : ptr(fs2, ra, rb),
}

local ceil = regentlib.ceil(float)
local log2 = regentlib.log2(float)
local log10 = regentlib.log10(float)
local log = regentlib.log(float)
local cos = regentlib.cos(float)
local sin = regentlib.sin(float)
local exp2 = regentlib.exp2(float)
local fabs = regentlib.fabs(float)
local trunc = regentlib.trunc(float)
local floor = regentlib.floor(float)
local sqrt = regentlib.sqrt(float)
local exp = regentlib.exp(float)

task f(ra : region(fs2), rb : region(fs2), r : region(fs1(ra, rb)))
where
  reads(r, ra, rb),
  writes(r, ra, rb)
do
  __demand(__vectorize)
  for e in r do
    var x = e.v1.x
    var y = e.v1.y
    var a = exp2(exp(log10(fabs(log2(fabs(log(fabs(sin(cos(fabs(sqrt(x + 10))))))))))))
    var b = exp2(exp(log10(fabs(log2(fabs(log(fabs(sin(cos(fabs(sqrt(y + 10))))))))))))
    e.p.v1.x += a
    e.p.v1.y += b
    e.v1.x -= a
    e.v1.y *= b
  end
  __forbid(__vectorize)
  for e in r do
    var x = e.v2.x
    var y = e.v2.y
    var a = exp2(exp(log10(fabs(log2(fabs(log(fabs(sin(cos(fabs(sqrt(x + 10))))))))))))
    var b = exp2(exp(log10(fabs(log2(fabs(log(fabs(sin(cos(fabs(sqrt(y + 10))))))))))))
    e.p.v2.x += a
    e.p.v2.y += b
    e.v2.x -= a
    e.v2.y *= b
  end
end

task toplevel()
  var n = 20
  var ra = region(ispace(ptr, n / 2), fs2)
  var rb = region(ispace(ptr, n / 2), fs2)
  var r1 = region(ispace(ptr, n), fs1(ra, rb))
  for i = 0, n do
    var ptra = dynamic_cast(ptr(fs2, ra), i / 2)
    if i % 2 == 0 then
      ptra.v1.x = 1.0 + i
      ptra.v1.y = 1.0 + i
      ptra.v2.x = 1.0 + i
      ptra.v2.y = 1.0 + i
    end

    var ptrb = dynamic_cast(ptr(fs2, rb), i / 2)
    if i % 2 == 0 then
      ptrb.v1.x = 4.0 + i
      ptrb.v1.y = 4.0 + i
      ptrb.v2.x = 4.0 + i
      ptrb.v2.y = 4.0 + i
    end

    var ptr1 = dynamic_cast(ptr(fs1(ra, rb), r1), i)
    ptr1.v1.x = 3.0 + i
    ptr1.v1.y = 3.0 + i
    ptr1.v2.x = 3.0 + i
    ptr1.v2.y = 3.0 + i

    if i % 2 == 0 then
      ptr1.p = static_cast(ptr(fs2, ra, rb), ptra)
    else
      ptr1.p = static_cast(ptr(fs2, ra, rb), ptrb)
    end
  end

  f(ra, rb, r1)

  for e in r1 do
    regentlib.assert(e.p.v1.x == e.p.v2.x, "test failed")
    regentlib.assert(e.p.v1.y == e.p.v2.y, "test failed")
    regentlib.assert(e.v1.x == e.v2.x, "test failed")
    regentlib.assert(e.v1.y == e.v2.y, "test failed")
  end
end

regentlib.start(toplevel)
