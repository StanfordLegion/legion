-- Copyright 2023 Stanford University, Los Alamos National Laboratory
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

import "regent"

local DEBUG = false

function make_fs(ty)
  local fspace wrap
  {
    v_add1 : ty;
    v_add2 : ty;
  }
  local fspace fs
  {
    input : ty;
    output_gpu : wrap;
    temp : wrap;
  }
  return fs
end

local types = terralib.newlist { uint32, int64 }
local formats = terralib.newlist { "%u", "%ld" }

local fs_types = {}
local format_strings = {}
for i = 1, #types do
  fs_types[types[i]] = make_fs(types[i])
  format_strings[types[i]] = formats[i]
end

function make_tasks(ty)
  local init_task, prefix_task, check_task
  local fs_type = fs_types[ty]

  __demand(__cuda)
  task init_task(r : region(ispace(int1d), fs_type))
  where
    reads writes(r)
  do
    for e in r do
      r[e].input = ([ty](e) + 5) % 10 + 1
      r[e].output_gpu.v_add1 = r[e].input
      r[e].output_gpu.v_add2 = r[e].input
      r[e].temp.v_add1 = [ty](0)
      r[e].temp.v_add2 = [ty](0)
    end
  end

  __demand(__cuda)
  task prefix_task(r : region(ispace(int1d), fs_type))
  where
    reads(r.input), reads writes(r.output_gpu)
  do
    var dir1 = 1
    var dir2 = -1
    __parallel_prefix(r.output_gpu.v_add1, r.output_gpu.v_add1, +, dir1)
    __parallel_prefix(r.output_gpu.v_add2, r.output_gpu.v_add2, +, dir2)
  end

  task check_task(r : region(ispace(int1d), fs_type))
  where
    reads(r.input), reads(r. output_gpu), reads writes(r.temp)
  do
    var bounds = r.bounds

    r[bounds.lo].temp.v_add1 = r[bounds.lo].input
    for i = [int64](bounds.lo) + 1, [int64](bounds.hi) + 1 do
      r[i].temp.v_add1 = r[i - 1].temp.v_add1 + r[i].input
    end

    r[bounds.hi].temp.v_add2 = r[bounds.hi].input
    for i = [int64](bounds.hi) - 1, [int64](bounds.lo) - 1, -1 do
      r[i].temp.v_add2 = r[i + 1].temp.v_add2 + r[i].input
    end

    for e in r do
      [(function()
        if DEBUG then return rquote
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. ", " .. format_strings[ty] .. "\n"],
              e.input, e.temp.v_add1, e.output_gpu.v_add1)
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. ", " .. format_strings[ty] .. "\n"],
              e.input, e.temp.v_add2, e.output_gpu.v_add2)
          end
        else
          return rquote end
        end
      end)()];
      regentlib.assert(e.temp.v_add1 == e.output_gpu.v_add1, "test failed")
      regentlib.assert(e.temp.v_add2 == e.output_gpu.v_add2, "test failed")
    end
  end

  return init_task, prefix_task, check_task
end

local init_tasks = {}
local prefix_tasks = {}
local check_tasks = {}

for i = 1, #types do
  init_tasks[types[i]], prefix_tasks[types[i]], check_tasks[types[i]]  = make_tasks(types[i])
end

local region_symbols = {}
for i = 1, #types do
  region_symbols[types[i]] = regentlib.newsymbol("r_" .. tostring(types[i]))
end

task test(size : int64)
  [types:map(function(ty)
    local fs = fs_types[ty]
    local sym = region_symbols[ty]
    local init = init_tasks[ty]
    local prefix = prefix_tasks[ty]
    local check = check_tasks[ty]
    return rquote
      var [sym] = region(ispace(int1d, size), fs)
      init([sym])
      prefix([sym])
      check([sym])
    end
  end)]
end

task main()
  for i = 1, 10, 3 do
    for j = 1, 10, 2 do
      for k = 1, 10 do
        test(256 * 256 * i + 256 * j + k)
        __fence(__execution, __block)
      end
    end
  end
end

regentlib.start(main)
