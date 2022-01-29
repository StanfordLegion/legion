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

-- This file is not meant to be run directly.

-- runs-with:
-- []

local args = rawget(_G, "arg")
local argc = #args
local argv = terralib.newsymbol((&int8)[argc + 2], "argv")
local argv_setup = terralib.newlist({quote var [argv] end})
for i = 0, argc do
  argv_setup:insert(quote
    [argv][ [i] ] = [ args[i] ]
  end)
end
argv_setup:insert(quote [argv][ [argc+1] ] = [&int8](0) end)

return { argv_setup = argv_setup, argc = argc+1, argv = argv }
