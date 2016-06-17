-- Copyright 2016 Stanford University
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

local args = rawget(_G, "arg")
local argc = #args
local argv = terralib.newsymbol((&int8)[argc], "argv")
local argv_setup = terralib.newlist({quote var [argv] end})
for i, arg in ipairs(args) do
  argv_setup:insert(quote
    [argv][ [i - 1] ] = [arg]
  end)
end

return { argv_setup = argv_setup, argc = argc, argv = argv }
