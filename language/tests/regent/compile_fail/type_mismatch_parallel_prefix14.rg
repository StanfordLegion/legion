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
-- type_mismatch_parallel_prefix14.rg:26: invalid privileges in argument 2: reads($s)
--   __parallel_prefix(r, s, +, 1)
--                        ^

import "regent"

task f(r : region(ispace(int1d, {5, 5}), double),
       s : region(ispace(int1d, {5, 5}), double))
where writes(r)
do
  __parallel_prefix(r, s, +, 1)
end
