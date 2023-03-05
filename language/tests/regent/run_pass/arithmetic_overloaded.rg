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
  x : double,
  y : double,
}

terra vec2.metamethods.__add(a : vec2, b : vec2) : vec2
  return vec2 { x = a.x + b.x, y = a.y + b.y }
end

terra vec2.metamethods.__sub(a : vec2, b : vec2) : vec2
  return vec2 { x = a.x - b.x, y = a.y - b.y }
end

vec2.metamethods.__mul = terralib.overloadedfunction(
  "__mul",
  {
    terra(a : double, b : vec2) : vec2
      return vec2 { x = a * b.x, y = a * b.y }
    end,
    terra(a : vec2, b : double) : vec2
      return vec2 { x = a.x * b, y = a.y * b }
    end
  })

task f(a : vec2, b : vec2) : vec2
  var c = a + b
  var d = a - b
  var e = 1.2 * a
  var f = e * 3.4
  return (c + d) - (e + f)
end

task main()
  var x = vec2 { x = 1, y = 2 }
  var y = vec2 { x = 30, y = 40 }
  f(x, y)
end
regentlib.start(main)
