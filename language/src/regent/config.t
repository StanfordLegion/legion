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

-- Legion Configuration and Command Line Parsing

local config = {}

local default_options = {
  ["aligned-instances"] = false,
  ["bounds-checks"] = false,
  ["cached-iterators"] = false,
  ["cuda"] = true,
  ["debug"] = false,
  ["no-dynamic-branches"] = true,
  ["no-dynamic-branches-assert"] = false,
  ["pretty"] = false,
  ["index-launches"] = true,
  ["futures"] = true,
  ["inlines"] = true,
  ["layout-constraints"] = true,
  ["leaf"] = true,
  ["trace"] = true,
  ["vectorize"] = true,
  ["validate"] = true,
  ["task-inlines"] = true,
  ["flow"] = true,
  ["flow-spmd"] = true,
  ["flow-spmd-shardsize"] = 1,
}

local option = {
  __index = function(t, k)
    error("no such option " .. tostring(k), 2)
  end,
  __newindex = function(t, k, v)
    error("options should only be set at startup time", 2)
  end,
}

function config.parse_args(rawargs)
  local options = {}
  for k, v in pairs(default_options) do
    options[k] = v
  end

  local args = terralib.newlist()

  if not rawargs then
    return setmetatable(options, option), args
  end

  local i = 0
  local arg_i = 1
  while rawargs[i] do
    local arg = rawargs[i]
    if string.sub(arg, 1, 2) == "-f" then
      local k = string.sub(rawargs[i], 3)
      if default_options[k] == nil then
        error("unknown option " .. rawargs[i])
      end
      if rawargs[i+1] == nil or tonumber(rawargs[i+1]) == nil then
        error("option " .. rawargs[i] .. " missing argument")
      end
      local v = tonumber(rawargs[i+1])
      if type(default_options[k]) == "boolean" then
        v = v~= 0
      end
      options[k] = v
      i = i + 1
    else
      args[arg_i] = rawargs[i]
      arg_i = arg_i + 1
    end
    i = i + 1
  end

  return setmetatable(options, option), args
end

local memoize_args = terralib.memoize(
  function()
    local rawargs = rawget(_G, "arg")
    return {config.parse_args(rawargs)}
  end)

function config.args()
  return unpack(memoize_args())
end

return config
