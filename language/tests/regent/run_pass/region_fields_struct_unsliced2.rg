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

struct vec2 {
  x : double;
  y : double;
}
vec2.__no_field_slicing = true

-- A collection of structs whose fields are not sliced
struct fs {
  ind   : int2d;
  rect  : rect2d;
  vec   : vec2;
  cmplx : complex;
}

task main()
  var is = ispace(int1d, 10)
  var r = region(is, fs)

  for e in r do
    e.ind = int2d { 0, 0 }
    e.ind.x  = e
    e.ind.y += e

    e.rect = rect2d { lo = {e, e}, hi = {e+1, e+1} }
    e.rect.lo += e.ind
    e.rect.hi.x += e.ind.x
    e.rect.hi.y  = e.ind.y

    e.vec = vec2 { 0.0, 0.0 }
    e.vec.x = 1.0
    e.vec.y += 2.0

    e.cmplx = complex { 0.0, 0.0 }
    e.cmplx.real = 1.0
    e.cmplx.imag += 2.0
  end

  for e in r do
    var v = [int64]([int1d](e))
    regentlib.assert(e.ind.x == v, "test failed")
    regentlib.assert(e.ind.y == v, "test failed")

    regentlib.assert(e.rect.lo.x == v + v, "test failed")
    regentlib.assert(e.rect.lo.y == v + v, "test failed")
    regentlib.assert(e.rect.hi.x == v + 1 + v, "test failed")
    regentlib.assert(e.rect.hi.y == v, "test failed")

    regentlib.assert(e.vec.x == 1.0, "test failed")
    regentlib.assert(e.vec.y == 2.0, "test failed")

    regentlib.assert(e.cmplx.real == 1.0, "test failed")
    regentlib.assert(e.cmplx.imag == 2.0, "test failed")
  end
end
regentlib.start(main)
