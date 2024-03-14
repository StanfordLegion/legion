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

local function make_check(r, t)
  local x = regentlib.newsymbol(ptr(int, r), "x")
  return rquote
    var [t] = 0
    for [x] in r do -- Region should be in scope for type of loop index.
      [t] += @x
    end
  end
end

local t = regentlib.newsymbol("t")
task main()
  var r = region(ispace(ptr, 5), int)
  fill(r, 0)

  for r : ptr(int, r) in r do -- Region should be in scope for type of loop index.
    @r += r
  end

  [make_check(r, t)]

  regentlib.assert([t] == 10, "test failed")
end
regentlib.start(main)
