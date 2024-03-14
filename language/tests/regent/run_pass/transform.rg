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

local function make_test(M, N)
  local tsk
  task tsk()
    var x : transform(M, N)
    for i = 0, M do
      for j = 0, N do
        x[{i, j}] = i + j
      end
    end
    for i = 0, M do
      for j = 0, N do
        regentlib.assert(x[{i, j}] == i + j, "test failed")
      end
    end
  end
  return tsk
end

local tests = terralib.newlist()
for i = 1, 3 do
  for j = 1, 3 do
    tests:insert(make_test(1, 3))
  end
end

task main()
  [tests:map(function(test) return rquote [test]() end end)]
end

regentlib.start(main)
