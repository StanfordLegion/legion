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

--[[--
Utilities for timing.

This module provides an interface for retrieving the current wall
clock time.

Two functions are provided, shown below. They return microseconds and
nanoseconds respectively, relative to the start of the Legion
runtime. Both functions return `int64`:

```lua
get_current_time_in_microseconds()
get_current_time_in_nanoseconds()
```

The implementation has been carefully designed to allow it to be
called from any kind of task. This makes it preferable to the C API,
which requires different implementations for leaf, inner and
replicable tasks.

@module std.timing
]]

import "regent"

local c = regentlib.c

local timing = {}

local function generate_get_time(unit_name, unit_short, unit_type)
  local get_time_op = c["legion_issue_timing_op_" .. unit_name]
  local get_time = c["legion_get_current_time_in_" .. unit_short]

  local terra get_time_future()
    -- We use the implicit context here to avoid tripping Regent's
    -- leaf optimization, which is ok because in a leaf task we will
    -- not issue an actual timing op but will use the direct method to
    -- retrieve time.
    var runtime = c.legion_runtime_get_runtime()
    var ctx = c.legion_runtime_get_context()
    var f : c.legion_future_t
    if c.legion_runtime_total_shards(runtime, ctx) > 1 then
      f = get_time_op(runtime, ctx)
    else
      var t : unit_type = get_time()
      f = c.legion_future_from_untyped_pointer(
        runtime, &t, terralib.sizeof(unit_type))
    end
    c.legion_context_destroy(ctx)
    return f
  end
  get_time_future.replicable = true

  local __demand(__inline) task get_time_task()
    var f = get_time_future()
    return __future(unit_type, f)
  end
  get_time_task:set_name("get_current_time_in_" .. tostring(unit_name))

  return get_time_task
end

timing.get_current_time_in_microseconds = generate_get_time(
  "microseconds", "micros", int64)

timing.get_current_time_in_nanoseconds = generate_get_time(
  "nanoseconds", "nanos", int64)

return timing
