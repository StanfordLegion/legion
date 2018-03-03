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

-- Logging

local config = require("common/config")

local log = {}

-- #####################################
-- ## Logging Level
-- #################

-- Shamelessly copying the Legion logging level hierarchy here...
local level = {
  ["spew"]    = 0,
  ["debug"]   = 1,
  ["info"]    = 2,
  ["print"]   = 3,
  ["warning"] = 4,
  ["error"]   = 5,
  ["fatal"]   = 6,
}

-- #####################################
-- ## Command-line Configuration
-- #################

local function parse_level(options, default_min_level)
  local line = options.log

  local matches = terralib.newlist()
  if string.len(line) > 0 then
    local position = 0
    while position do
      local new_position = string.find(line, ",", position+1)
      matches:insert(string.sub(line, position+1, new_position and new_position-1))
      position = new_position
    end
  end

  local min_level = default_min_level
  local categories = {}
  for _, match in ipairs(matches) do
    local position = string.find(match, "=")
    if position then
      local category = string.sub(match, 1, position-1)
      local level = tonumber(string.sub(match, position+1))
      assert(string.len(category) > 0 and level)
      categories[category] = level
    else
      local level = tonumber(match)
      assert(level)
      min_level = level
    end
  end
  return min_level, categories
end

local default_options = terralib.newlist({
  config.make_default_option("-flog", "log", "string", ""),
  config.make_default_option("-fdebug", "debug", "boolean", false),
})
local options, _ = config.args(default_options)

local default_min_level = (options.debug and level.info) or level.print
local min_level, categories = parse_level(options, default_min_level)

-- #####################################
-- ## Loggers
-- #################

local logger = {}
function logger:__index(field)
  local value = logger[field]
  if value ~= nil then return value end
  error("logger has no field '" .. field .. "' (in lookup)", 2)
end

function logger:__newindex(field, value)
  error("logger has no field '" .. field .. "' (in assignment)", 2)
end

local nop = function() end

function log.make_logger(category)
  local result = { category = category }

  local category_min_level = min_level
  if categories[category] then
    category_min_level = categories[category]
  end

  -- Replace disabled levels with nops.
  for name, value in pairs(level) do
    if value < category_min_level then
      result[name] = nop
    end
  end

  return setmetatable(result, logger)
end

function log.get_log_level(category)
  return categories[category] or min_level
end

function logger:log(level, format_string, ...)
  assert(type(level) == "number")
  assert(type(format_string) == "string")
  io.stderr:write(string.format("{%d}{%s}: ", level, self.category) ..
                  string.format(format_string, ...) .. "\n")
end

for name, value in pairs(level) do
  logger[name] = function(self, ...)
    self:log(value, ...)
  end
end

return log
