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

import "regent"

do
  -- Symbols can be untyped, so each occurance should be allowed to
  -- unify to a different type. This requires the compiler to copy the
  -- symbols on specialization.
  local x = regentlib.newsymbol("x")
  local decl1 = rquote do var [x] = 0 if [x] ~= 0 then return 7 end end end
  local decl2 = rquote do var [x] = true if not [x] then return 18 end end end
  x = false -- Specialization must happen before here.
  task t1()
    [decl1];
    [decl2]
    return 4
  end
end

do
  -- It should also be possible to write non-heigenic code where
  -- multiple statements are spliced into a single scope. These
  -- occurances of x need to be the same symbol.
  local x = regentlib.newsymbol("x")
  local decl1 = rquote var [x] = 1234 end
  local expr1 = rexpr x + 1 end
  local use1 = rquote
    [x] = x * 2 -- Both [x] and x should work.
    do
      var [x] = [x] + 10000 -- Nested scopes should also work.
      if x > 0 then return [expr1] end
    end
  end
  x = false -- Specialization must happen before here.
  task t2()
    [decl1];
    [use1]
    return 1001
  end
end

task main()
  regentlib.assert(t1() == 4, "test failed")
  regentlib.assert(t2() == 12469, "test failed")
end
regentlib.start(main)
