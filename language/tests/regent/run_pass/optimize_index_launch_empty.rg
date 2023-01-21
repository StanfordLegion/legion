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

import "regent"

local format = require("std/format")

-- This tests a bug resulting from empty index launches and reductions
-- on futures.

local task foo()
  return 1.0
end

local task main()
  var acc = 0.0
  var empty = ispace(int1d, 0)
  for t in empty do
    acc += foo()
  end
  var res = acc/12
  format.println("{}", res)
end
regentlib.start(main)
