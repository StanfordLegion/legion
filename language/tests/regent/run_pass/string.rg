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

import "regent"

local cstdio = terralib.includec("stdio.h")

task needs_string(x : regentlib.string)
  cstdio.printf("from subtask: %s", [rawstring](x))
end

task main()
  -- A Terra rawstring
  var x = "hello, rawstring!\n"
  cstdio.printf(x)

  -- A Regent string
  var y : regentlib.string = "hello, string!\n"
  cstdio.printf([rawstring](y))

  needs_string(y)
end
regentlib.start(main)
