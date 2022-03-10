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

  local terra get_current_time()
    -- We use the implicit context here to avoid tripping Regent's
    -- leaf optimization, which is ok because in a leaf task we will
    -- not issue an actual timing op but will use the direct method to
    -- retrieve time.
    var runtime = c.legion_runtime_get_runtime()
    var ctx = c.legion_runtime_get_context()
    if c.legion_runtime_total_shards(runtime, ctx) > 1 then
      var f = get_time_op(runtime, ctx)
      regentlib.assert(
        c.legion_future_get_untyped_size(f) == terralib.sizeof(unit_type),
        ["unexpected future size in get_current_time_in_" .. unit_name])
      c.legion_context_destroy(ctx)
      return @([&unit_type](c.legion_future_get_untyped_pointer(f)))
    else
      c.legion_context_destroy(ctx)
      return get_time()
    end
  end
  get_current_time.replicable = true

  return get_current_time
end

timing.get_current_time_in_microseconds = generate_get_time(
  "microseconds", "micros", int64)

timing.get_current_time_in_nanoseconds = generate_get_time(
  "nanoseconds", "nanos", int64)

return timing
