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
--  ["-fjobs", "2"]
-- ]

import "regent"

task foo(x : int) : int
  return x
end

task main()
  regentlib.assert(foo(42) == 42, "test failed")
end

local exe = os.tmpname()
regentlib.saveobj(main, exe, "executable")

local env = ""
if os.getenv("DYLD_LIBRARY_PATH") then
  env = "DYLD_LIBRARY_PATH='" .. os.getenv("DYLD_LIBRARY_PATH") .. "' "
elseif os.getenv("LD_LIBRARY_PATH") then
  env = "LD_LIBRARY_PATH='" .. os.getenv("LD_LIBRARY_PATH") .. "' "
end

-- Pass the arguments along so that the child process is able to
-- complete the execution of the parent.
local args = ""
for _, arg in ipairs(rawget(_G, "arg")) do
  args = args .. " '" .. arg .. "'"
end
assert(os.execute(env .. exe .. args) == 0)
