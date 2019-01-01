-- Copyright 2019 Stanford University
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

-- fails-with:
-- invalid_multiple_field_access4.rg:41: invalid use of multi-field access
--   v2.{x, y} = f:apply(v1.{x, y})
--    ^

import "regent"

struct adder
{
  inc : int,
}

terra adder:apply(x : int)
  return self.inc + x
end

struct vec2
{
  x : int,
  y : int,
}

task main()
  var f : adder = adder { inc = 10 }
  var v1 : vec2, v2 : vec2
  v1 = vec2 { x = 1, y = 2 }
  v2.{x, y} = f:apply(v1.{x, y})
  -- use these instead
  --v2.x = f:apply(v1.x)
  --v2.y = f:apply(v1.y)
end
regentlib.start(main)
