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
-- invalid_multiple_field_access2.rg:35: invalid use of multi-field access
--   c.x = a.{z, y, x}
--   ^

import "regent"

struct foo
{
  x : int,
  y : int,
  z : int,
}

task f() : double
  var a : foo, c : foo
  a.x, a.y, a.z = 2, 1, 3
  -- this statement is valid
  a.{z, y, x} = c.x
  -- this is not
  c.x = a.{z, y, x}
end

task main()
  f()
end
regentlib.start(main)
