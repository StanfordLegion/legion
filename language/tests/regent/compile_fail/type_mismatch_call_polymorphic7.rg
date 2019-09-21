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
-- type_mismatch_call_polymorphic7.rg:48: type mismatch: expected int32 for field b but got double
--     init_double(p[c].{v})
--                ^

import "regent"

struct vec2
{
  _x : double;
  _y : double;
}

fspace fs
{
  i : int;
  v : vec2;
}

struct iface
{
  a : double;
  b : int;
}

task init_double(x : region(ispace(int1d), iface)) end

task main()
  var r = region(ispace(int1d, 5), fs)
  var p = partition(equal, r, ispace(int1d, 2))

  __demand(__index_launch)
  for c in p.colors do
    init_double(p[c].{v})
  end
end
