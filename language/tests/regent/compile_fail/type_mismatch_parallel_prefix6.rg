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

-- fails-with:
-- type_mismatch_parallel_prefix6.rg:37: type mismatch in argument 2: expected a primitive type, but got fs2()
--     __parallel_prefix(r.v1, s.v2, +, 1)
--                             ^

import "regent"

fspace fs2
{
  x : double,
  y : double,
}

fspace fs
{
  v1 : double,
  v2 : fs2,
}

task f()
  var r = region(ispace(int1d, 5), fs)
  var s = region(ispace(int1d, 5), fs)
  __parallel_prefix(r.v1, s.v2, +, 1)
end
