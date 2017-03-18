-- Copyright 2017 Stanford University
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

-- Regent Configuration and Command Line Parsing

local common_config = require("common/config")

local config = {}

local default_options = {
  -- Main user-facing correctness flags:
  ["bounds-checks"] = false,

  -- Main user-facing optimization flags:
  ["cuda"] = true,
  ["index-launch"] = true,
  ["inline"] = true,
  ["future"] = true,
  ["leaf"] = true,
  ["inner"] = true,
  ["mapping"] = true,
  ["vectorize"] = true,
  ["vectorize-unsafe"] = false,

  -- Legion runtime optimization flags:
  ["legion-leaf"] = true,
  ["legion-inner"] = true,

  -- Dataflow optimization flags:
  ["flow"] = os.getenv('USE_RDIR') == '1' or false,
  ["flow-spmd"] = false,
  ["flow-spmd-shardsize"] = 1,

  -- Experimental auto-parallelization flags:
  ["parallelize"] = true,
  ["parallelize-dop"] = "4",
  ["parallelize-global"] = true,

  -- Miscellaneous, internal or special-purpose flags:
  ["aligned-instances"] = false,
  ["cached-iterators"] = false,
  ["debug"] = false,
  ["no-dynamic-branches"] = true,
  ["no-dynamic-branches-assert"] = false,
  ["pretty"] = false,
  ["layout-constraints"] = true,
  ["trace"] = true,
  ["validate"] = true,
  ["emergency-gc"] = false,

  -- Need this here to make the logger happy.
  ["log"] = "",
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
    make_default_options("-f", default_options),
    "-f")
end

return config
