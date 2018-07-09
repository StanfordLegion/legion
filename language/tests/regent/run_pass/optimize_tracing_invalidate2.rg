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

-- runs-with:
-- [
--  [ "-dm:memoize" ],
--  [ "-dm:memoize", "-lg:no_fence_elision" ],
--  [ "-dm:memoize", "-lg:no_trace_optimization" ]
-- ]

import "regent"
import "bishop"

mapper

$PROCS = processors[isa=x86]

task#foo, task#bar
{
  target : $PROCS[0 % $PROCS.size];
}

end

local c = regentlib.c

task foo(r : region(ispace(int1d), int))
where reads writes(r) do return 1 end

task bar(r : region(ispace(int1d), int))
where reads writes(r) do end

task toplevel()
  var r = region(ispace(int1d, 5), int)
  var lr = __raw(r)
  var field = __fields(r)[0]

  for i = 0, 10 do
    var il =
      c.legion_inline_launcher_create_logical_region(lr, c.READ_WRITE, c.EXCLUSIVE, lr,
                                                     0, false, 0, 0)
    c.legion_inline_launcher_add_field(il, field, true)
    var pr = c.legion_inline_launcher_execute(__runtime(), __context(), il)
    c.legion_inline_launcher_destroy(il)
    __demand(__trace)
    do
      c.legion_physical_region_wait_until_valid(pr)
      c.legion_runtime_unmap_region(__runtime(), __context(), pr)
      foo(r)
      bar(r)
    end
  end

  __demand(__trace)
  for i = 0, 10 do
    var x = foo(r)
    if x == 1 then bar(r) end
  end
end

regentlib.start(toplevel, bishoplib.make_entry())
