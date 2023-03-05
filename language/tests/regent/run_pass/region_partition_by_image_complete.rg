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

struct t {
  value : int,
  color : int1d,
}

task f()
  var r = region(ispace(ptr, 3), t)

  do
    var i = 0
    for x in r do
      x.value = 0
      x.color = i
      i += 1
    end
  end

  var p = partition(incomplete, r.color, ispace(int1d, 3))
  -- This file is identical to the region_partition_by_field.rg except these line, which add a
  -- (hopefully) well-formed call to image. Our goal is to test every permuatation of disjointness
  -- and completeness here
  var img = image(r, p, r.color)
  var img2 = image(aliased, incomplete, r, p, r.color)

  for i = 0, 3 do
    var ri = p[i]
    for x in ri do
      x.value = (1 + i) * (1 + i)
    end
  end

  var s = 0
  for i = 0, 3 do
    var ri = p[i]
    for x in ri do
      s += x.value
    end
  end

  return s
end

task main()
  regentlib.assert(f() == 14, "test failed")
end
regentlib.start(main)
