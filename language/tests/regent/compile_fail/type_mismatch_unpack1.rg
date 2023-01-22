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
-- type_mismatch_unpack1.rg:32: type mismatch in argument 2: expected ptr(int32, $d) but got ptr(int32, $q)
--   f(d, e)
--   ^

import "regent"

fspace s {
  q : region(int),
  t : ptr(int, q),
}

task f(a : region(int), b : ptr(int, a)) end

task g(c : s)
  var d = c.q
  var e = c.t
  f(d, e)
end
g:compile()
