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

-- fails-with:
-- invalid_fill2.rg:35: partial fill with type fs2 is not allowed
--   fill(r.val.real, 0)
--      ^

import "regent"

struct fs2
{
  real : double,
}
fs2.__no_field_slicing = true

struct fs1
{
  val : fs2,
}

task k(r : region(fs1))
where writes(r) do
  fill(r.val.real, 0)
end
