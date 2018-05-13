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

task foo(r : region(ispace(int1d), int))
where reads writes(r) do return 1 end

task bar(r : region(ispace(int1d), int))
where reads writes(r) do end

task toplevel()
  var r = region(ispace(int1d, 5), int)

  __demand(__trace)
  for i = 0, 10 do
    foo(r)
    bar(r)
    for e in r do r[e] = 1 end
  end

  __demand(__trace)
  for i = 0, 10 do
    var x = foo(r)
    if x == 1 then bar(r) end
  end
end

regentlib.start(toplevel, bishoplib.make_entry())
