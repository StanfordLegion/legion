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

import "regent"

local format = require("std/format")

fspace elt {
  a : int,
  b : int,
}

__demand(__inline)
task blur(a : int, b : int, c : int)
  return (a + b + c)/3 + 1
end

__demand(__leaf)
task stencil(r : region(ispace(int1d), elt),
             r_left : region(ispace(int1d), elt),
             r_right : region(ispace(int1d), elt),
             r_interior : region(ispace(int1d), elt),
             bounds : rect1d)
where reads writes(r.b), reads(r.a, r_left.a, r_right.a) do
  do
    var x = r.bounds.lo
    r[x].b = blur(r[x].a, r_left[(x-1)%bounds].a, r[(x+1)%bounds].a)
  end
  do
    var x = r.bounds.hi
    r[x].b = blur(r[x].a, r[(x-1)%bounds].a, r_right[(x+1)%bounds].a)
  end
  for x in r_interior do
    r[x].b = blur(r[x].a, r[(x-1)%bounds].a, r[(x+1)%bounds].a)
  end
end

__demand(__leaf)
task interior(bounds : rect1d) : rect1d
  return rect1d { bounds.lo+1, bounds.hi-1}
end

-- FIXME: Need to avoid hard-coding bounds here.
__demand(__leaf)
task halo_left(bounds : rect1d) : rect1d
  var parent = rect1d { 0, 1023 }
  return rect1d { (bounds.lo-1)%parent, (bounds.lo-1)%parent}
end

__demand(__leaf)
task halo_right(bounds : rect1d) : rect1d
  var parent = rect1d { 0, 1023 }
  return rect1d { (bounds.hi+1)%parent, (bounds.hi+1)%parent }
end

__demand(__replicable, __inner)
task main()
  format.println("Main running...")
  var num_elts = 1024
  var num_colors = 16
  var colors = ispace(int1d, num_colors)

  var r = region(ispace(int1d, num_elts), elt)
  var p = partition(equal, r, colors)
  var p_interior = image(r, p, interior)
  var p_left = image(r, p, halo_left)
  var p_right = image(r, p, halo_right)

  var bounds = r.bounds

  fill(r.{a, b}, 0)

  __demand(__constant_time_launch)
  for i in colors do
    stencil(p[i], p_left[i], p_right[i], p_interior[i], bounds)
  end
  format.println("Main complete.")
end
regentlib.start(main)
