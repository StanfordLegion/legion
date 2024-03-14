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

-- This tests a bug where the compiler wasn't registering references
-- to external names properly, and the symbols were showing up as
-- undefined.

local arg1 = regentlib.newsymbol(region(ispace(int1d),int),'x')
local arg2 = regentlib.newsymbol(region(ispace(int1d),int),'y')

local two_args = terralib.newlist()
two_args:insert(arg1)
two_args:insert(arg2)

task good([arg1], [arg2]) where arg1 <= arg2 do end

task bad([two_args]) where arg1 <= arg2 do end

task main()
  var r = region(ispace(int1d, 5), int)
  var p = partition(equal, r, ispace(int1d, 1))
  var r0 = p[0]

  good(r0, r)
  bad(r0, r)
end
regentlib.start(main)
