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

-- Bishop Configuration and Command Line Parsing

local common_config = require("common/config")

local config = {}

local default_options = {
  ["standalone"] = false,
  ["taskid-map"] = "",
  ["dump-dfa"] = "",
}

local function make_default_options(prefix, options)
  local result = terralib.newlist()
  for k, v in pairs(options) do
    result:insert(
      common_config.make_default_option(prefix .. k, k, type(v), v))
  end
  return result
end

function config.args()
  return common_config.args(
    make_default_options("-bishop:", default_options),
    "-bishop:")
end

return config
