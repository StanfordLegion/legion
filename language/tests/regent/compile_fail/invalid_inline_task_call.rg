-- Copyright 2018 Stanford University
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
-- invalid_inline_task_call.rg:51: cannot inline a task that does not demand inlining
--   var ret_f = __demand(__inline, f(r, x))
--                                     ^

import "regent"

local c = regentlib.c

struct ret
{
  v : int,
  id : uint64,
}

task f(s : region(int), y : ptr(int, s)) : ret
where reads(s) do
  return ret { v = @y, id = c.legion_context_get_unique_id(__context()) }
end

task h(s : region(int), y : ptr(int, s)) : ret
where reads(s), writes(s) do
  @y += 100
  return ret { v = @y, id = c.legion_context_get_unique_id(__context()) }
end

__demand(__inline)
task id(x : int) : int
  return x
end

task g() : ret
  var id_g = c.legion_context_get_unique_id(__context())
  var r = region(ispace(ptr, 5), int)
  var x = dynamic_cast(ptr(int, r), 0)
  @x = 5
  var ret_f = __demand(__inline, f(r, x))
  regentlib.assert(id_g == ret_f.id, "test failed")
  var ret_h = __demand(__inline, h(r, x))
  regentlib.assert(id_g == ret_h.id, "test failed")
  f(r, x)
  for i = 0, 5 do
    h(r, x)
  end
  return __demand(__inline, f(r, x))
end

task main()
  var ret_g = g()
  regentlib.assert(ret_g.v == 605, "test failed")
end
regentlib.start(main)
