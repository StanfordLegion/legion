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

-- Profiling compiler passes

local log = require("common/log")
local log_profile = log.make_logger("profile")

local profile = {}

function profile:__index(field)
  error("profiler has no field '" .. field .. "' (in lookup)", 2)
end

function profile:__newindex(field, value)
  error("profiler has no field '" .. field .. "' (in assignment)", 2)
end

if log.get_log_level("profile") >= 3 then
  function profile:__call(pass_name, node, fn)
    return fn
  end

else
  local ast = require("regent/ast")
  local std = require("regent/std")
  local c = std.c

  function profile:__call(pass_name, node, fn)
    assert(pass_name)
    assert(node)
    assert(fn)
    return function(...)
      local start = tonumber(c.legion_get_current_time_in_micros())
      local result = {fn(...)}
      local stop = tonumber(c.legion_get_current_time_in_micros())
      if node:is(ast.unspecialized.top.Task) or
         node:is(ast.specialized.top.Task) or
         node:is(ast.typed.top.Task) then
        log_profile:info(tostring(node.name) .. " " ..
          pass_name .. " took " ..
          tostring((stop - start) * 1e-6) .. "s")
      end
      return unpack(result)
    end
  end
end

return setmetatable({}, profile)
