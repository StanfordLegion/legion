-- Copyright 2022 Stanford University
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

local c = regentlib.c

task hello()
  var args = c.legion_runtime_get_input_args()
  regentlib.assert(args.argc > 0, "no args")
  format.println("hello world from {}", args.argv[0])
end

task main()
  hello()
end

local exe = os.tmpname()
regentlib.saveobj(main, exe, "executable")
print("Saved executable to " .. exe)

-- Hack: On macOS, the child process isn't inheriting the parent's
-- environment, for some reason.
local env = ""
if os.getenv("DYLD_LIBRARY_PATH") then
  env = "DYLD_LIBRARY_PATH=" .. os.getenv("DYLD_LIBRARY_PATH") .. " "
end

-- Pass the arguments along so that the child process is able to
-- complete the execution of the parent.
local args = ""
for _, arg in ipairs(rawget(_G, "arg")) do
  args = args .. " " .. arg
end

assert(os.execute(env .. exe .. args) == 0)
