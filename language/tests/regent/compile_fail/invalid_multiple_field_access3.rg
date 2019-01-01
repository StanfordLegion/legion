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
-- invalid_multiple_field_access3.rg:36: invalid use of multi-field access
--   b.{x, y, z} = f(a.{x, y, z})
--   ^

import "regent"

struct foo
{
  x : int,
  y : int,
  z : int,
}

task f(x : int) : int
  return x + 10
end

task main()
  var a : foo, b : foo
  a = foo { x = 1, y = 2, z = 3 }
  b.{x, y, z} = f(a.{x, y, z})
end
regentlib.start(main)
