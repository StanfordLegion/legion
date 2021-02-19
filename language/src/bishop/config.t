-- Copyright 2021 Stanford University
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

function config.args()
  local options = terralib.newlist()
  common_config.add_default_options("-bishop:", default_options, options)
  return common_config.args(options, "-bishop:")
end

return config
