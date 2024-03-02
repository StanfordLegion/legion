-- Copyright 2024 Stanford University, Los Alamos National Laboratory
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

import "regent"

local c = regentlib.c

fspace fs1
{
  pad1 : int;
  data : int[10];
  pad2 : int;
}

fspace fs2
{
  pad1 : int;
  data : regentlib.array(int, 10);
  pad2 : int;
}

function make_task(ty)
  local tsk
  task tsk(is : ispace(int1d),
           r : region(is, ty))
  where reads writes(r) do
    for p in is do
      var data = r[p].data
      r[p].data = data
      for i = 0, 10 do
        r[p].data[i] += data[i]
      end
      for i = 0, 10 do
        r[p].data[i] += 1
        r[p].data[i] = r[p].data[i] + 2
      end
    end
  end
  return tsk
end
local foo = make_task(fs1)
local bar = make_task(fs2)

task test_conversion(is : ispace(int1d), r : region(is, fs2))
where reads writes(r) do
  for p in is do
    var tmp : fs2 = r[p]
    for i = 0, 10 do
      regentlib.assert(tmp.data[i] == 2 * (100 + [int](p)) + 3, "test failed")
      tmp.data[i] += 10
      r[p] = tmp
      regentlib.assert(r[p].data[i] == 2 * (100 + [int](p)) + 13, "test failed")
    end
  end
end

task toplevel()
  var is = ispace(int1d, 10)
  var r = region(is, fs1)
  var s = region(is, fs2)

  for p in is do
    for i = 0, 10 do
      r[p].data[i] = 100 + [int](p)
      s[p].data[i] = 100 + [int](p)
    end
  end

  foo(is, r)
  bar(is, s)

  for p in is do
    for i = 0, 10 do
      regentlib.assert(r[p].data[i] == 2 * (100 + [int](p)) + 3, "test failed")
      regentlib.assert(s[p].data[i] == 2 * (100 + [int](p)) + 3, "test failed")
    end
  end

  test_conversion(is, s)
end

regentlib.start(toplevel)
