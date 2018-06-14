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

-- Profiling compiler passes

local data = require("common/data")
local log = require("common/log")

local log_profile = log.make_logger("profile")

local profile = {}

function profile:__index(field)
  local value = rawget(profile, field)
  if value then
    return value
  end
  error("profiler has no field '" .. field .. "' (in lookup)", 2)
end

function profile:__newindex(field, value)
  error("profiler has no field '" .. field .. "' (in assignment)", 2)
end

local first_import_time
local total_time_by_pass = data.new_default_map(function() return 0 end)

if log.get_log_level("profile") >= 3 then
  function profile:__call(pass_name, node, fn)
    return fn
  end

  function profile.set_import_time()
  end

  function profile.print_summary()
  end
else
  local ast = require("regent/ast")
  local base = require("regent/std_base")
  local c = base.c

  function profile:__call(pass_name, target, fn)
    assert(pass_name)
    assert(fn)
    return function(...)
      local start = tonumber(c.legion_get_current_time_in_micros())
      local result = {fn(...)}
      local stop = tonumber(c.legion_get_current_time_in_micros())
      local elapsed = (stop - start) * 1e-6
      local tag
      if not target then
        tag = ""
      elseif base.is_variant(target) then
        tag = tostring(target) .. " "
      elseif ast.is_node(target) and (target:is(ast.unspecialized.top.Task) or
                                      target:is(ast.specialized.top.Task) or
                                      target:is(ast.typed.top.Task)) then
        tag = tostring(target.name) .. " "
      end
      if tag then
        log_profile:info(tag .. pass_name .. " took " ..
	                 tostring(elapsed) .. " s")

      end
      total_time_by_pass[pass_name] = total_time_by_pass[pass_name] + elapsed
      return unpack(result)
    end
  end

  function profile.set_import_time()
    assert(first_import_time == nil)
    first_import_time = tonumber(c.legion_get_current_time_in_micros())
  end

  function profile.print_summary()
    local final_time = tonumber(c.legion_get_current_time_in_micros())
    local total_elapsed = (final_time - first_import_time) * 1e-6

    local pass_times = terralib.newlist()
    for k, v in total_time_by_pass:items() do
      pass_times:insert({k, v})
    end
    table.sort(pass_times, function(a, b) return a[2] < b[2] end)
    log_profile:info("########################################")
    log_profile:info("##  summary of total execution time   ##")
    log_profile:info("########################################")
    local total_pass_time = 0
    for _, kv in ipairs(pass_times) do
      local k, v = unpack(kv)
      log_profile:info(k .. " " .. tostring(v) .. " s")
      total_pass_time = total_pass_time + v
    end
    log_profile:info("total " .. tostring(total_elapsed) .. " s")
    log_profile:info("unaccounted " .. tostring(total_elapsed - total_pass_time) .. " s")
  end
end

return setmetatable({}, profile)
