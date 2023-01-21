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
-- privilege_field_access5.rg:32: invalid privilege reads($r.b) for dereference of ptr(t, $r)
--   var x = p.b
--           ^

import "regent"

struct s {
  a : int
}

struct t {
  b : s,
}

task f(r : region(t), p : ptr(t, r)) : int
where reads(r.b.a) do
  var x = p.b
  return x.a
end
f:compile()
