-- Copyright 2015 Stanford University, NVIDIA Corporation
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

-- Bishop Language Entry Point

local parser = require("bishop/parser")
local specialize = require("bishop/specialize")

local language = {
  name = "bishop",
  entrypoints = {
    "bishop",
  },
  keywords = {
    -- elements
    "task",
    "region",
    "for",
    "while",
    "do",

    -- parameter
    "parameter",

    -- trigger
    "trigger",
    "when",

    "bishop",
    "end",
  },
}

function language:statement(lex)
  local node = parser:parse(lex)
  local function ctor(environment_function)
    return node
  end
  return ctor, {"__bishop__"}
end

return language
