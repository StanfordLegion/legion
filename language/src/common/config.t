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

-- Command Line Parsing

local data = require("common/data")

local config = {}

local default_option = {
  __index = function(t, k)
    error("default option has no field " .. tostring(k), 2)
  end,
  __newindex = function(t, k, v)
    error("default option is not modifiable", 2)
  end,
}

local supported_flag_types = {
  ["number"] = true,
  ["boolean"] = true,
  ["string"] = true,
}

function config.make_default_option(flag, name, flag_type, value)
  assert(type(flag) == "string")
  assert(type(name) == "string")
  assert(supported_flag_types[flag_type])
  assert(type(flag_type) == "string")
  return setmetatable(
    {
      flag = flag,
      name = name,
      type = flag_type,
      value = value,
    },
    default_option)
end

function config.is_default_option(opt)
  return getmetatable(opt) == default_option
end

local option_set = {
  __index = function(t, k)
    error("no such option " .. tostring(k), 2)
  end,
  __newindex = function(t, k, v)
    error("options should only be set at startup time", 2)
  end,
}

function config.parse_args(default_options, prefix, rawargs)
  local index = {}
  for _, opt in ipairs(default_options) do
    index[opt.flag] = opt
  end

  local options = {}
  for _, opt in ipairs(default_options) do
    options[opt.name] = opt.value
  end

  local args = terralib.newlist()

  if not rawargs then
    return setmetatable(options, option_set), args
  end

  local i = 0
  local arg_i = 1
  while rawargs[i] do
    local arg = rawargs[i]
    local opt = index[arg]
    if opt then
      if rawargs[i+1] == nil then
        error("option " .. arg .. " missing argument")
      end

      local v
      if opt.type == "number" then
        v = tonumber(rawargs[i+1])
        if v == nil then
          error("option " .. arg .. " missing argument")
        end
      elseif opt.type == "boolean" then
        v = tonumber(rawargs[i+1])
        if v == nil then
          error("option " .. arg .. " missing argument")
        end
        v = v ~= 0
      elseif opt.type == "string" then
        v = rawargs[i+1]
      else
        assert(false)
      end

      options[opt.name] = v
      i = i + 1
    elseif prefix and string.sub(arg, 1, string.len(prefix)) == prefix then
      error("unknown option " .. arg)
    else
      args[arg_i] = arg
      arg_i = arg_i + 1
    end
    i = i + 1
  end

  return setmetatable(options, option_set), args
end

local memoize_args = terralib.memoize(
  function(default_options, prefix)
    local rawargs = rawget(_G, "arg")
    return {config.parse_args(default_options, prefix, rawargs)}
  end)

function config.args(default_options, prefix)
  assert(terralib.islist(default_options) and
           data.all(unpack(default_options:map(config.is_default_option))),
         "expected a list of default options")
  return unpack(memoize_args(default_options, prefix))
end

return config
