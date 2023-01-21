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
-- type_mismatch_parallel_prefix4.rg:31: type mismatch in argument 2: expected one or no field path, but got 2
--   __parallel_prefix(r.v1, s.{v1, v2}, +, 1)
--                             ^

import "regent"

fspace fs
{
  v1 : double,
  v2 : double,
}

task f()
  var r = region(ispace(int1d, 5), fs)
  var s = region(ispace(int1d, 5), fs)
  __parallel_prefix(r.v1, s.{v1, v2}, +, 1)
end
