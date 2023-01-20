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

-- runs-with:
-- [
--   ["-fbounds-checks", "0"],
--   ["-fbounds-checks", "1", "-fcuda", "0"]
-- ]

import "regent"

local DEBUG = false

function make_fs(ty)
  local fspace wrap
  {
    v_add : ty;
    v_mul : ty;
    v_min : ty;
    v_max : ty;
  }
  local fspace fs
  {
    input : ty;
    output_cpu : wrap;
    output_gpu : wrap;
    temp : wrap;
  }
  return fs
end

local types = terralib.newlist { double, uint32, int64 }
local formats = terralib.newlist { "%lf", "%u", "%ld" }

local fs_types = {}
local format_strings = {}
for i = 1, #types do
  fs_types[types[i]] = make_fs(types[i])
  format_strings[types[i]] = formats[i]
end

function make_tasks(ty)
  local init_task, cpu_prefix_task, gpu_prefix_task, check_task
  local fs_type = fs_types[ty]

  task init_task(r : region(ispace(int1d), fs_type))
  where
    reads writes(r)
  do
    for e in r do
      r[e].input = ([ty](e) + 5) % 10 + 1
      r[e].output_cpu.v_add = [ty](0)
      r[e].output_cpu.v_mul = [ty](0)
      r[e].output_cpu.v_min = [ty](0)
      r[e].output_cpu.v_max = [ty](0)
      r[e].output_gpu.v_add = [ty](0)
      r[e].output_gpu.v_mul = [ty](0)
      r[e].output_gpu.v_min = [ty](0)
      r[e].output_gpu.v_max = [ty](0)
      r[e].temp.v_add = [ty](0)
      r[e].temp.v_mul = [ty](0)
      r[e].temp.v_min = [ty](0)
      r[e].temp.v_max = [ty](0)
    end
  end

  task cpu_prefix_task(r : region(ispace(int1d), fs_type))
  where
    reads(r.input), reads writes(r.output_cpu)
  do
    var dir1 = 1
    var dir2 = -1
    __parallel_prefix(r.output_cpu.v_add, r.input,              +, dir1)
    __parallel_prefix(r.output_cpu.v_mul, r.output_cpu.v_add,   *, dir2)
    -- TODO: Put these tests back once this pull request is merged and
    --       Regent picks up this change:
    --         https://github.com/zdevito/terra/pull/302
    --
    -- __parallel_prefix(r.output_cpu.v_min, r.output_cpu.v_mul, min,  1)
    -- __parallel_prefix(r.output_cpu.v_max, r.output_cpu.v_min, max, -1)
  end

  __demand(__cuda)
  task gpu_prefix_task(r : region(ispace(int1d), fs_type))
  where
    reads(r.input), reads writes(r.output_gpu)
  do
    var dir1 = 1
    var dir2 = -1
    __parallel_prefix(r.output_gpu.v_add, r.input,              +, dir1)
    __parallel_prefix(r.output_gpu.v_mul, r.output_gpu.v_add,   *, dir2)
    -- TODO: Put these tests back once this pull request is merged and
    --       Regent picks up this change:
    --         https://github.com/zdevito/terra/pull/302
    --
    -- __parallel_prefix(r.output_gpu.v_min, r.output_gpu.v_mul, min,  1)
    -- __parallel_prefix(r.output_gpu.v_max, r.output_gpu.v_min, max, -1)
  end

  task check_task(r : region(ispace(int1d), fs_type))
  where
    reads(r.input), reads(r.{output_cpu, output_gpu}), reads writes(r.temp)
  do
    var bounds = r.bounds

    r[bounds.lo].temp.v_add = r[bounds.lo].input
    for i = [int64](bounds.lo) + 1, [int64](bounds.hi) + 1 do
      r[i].temp.v_add = r[i - 1].temp.v_add + r[i].input
    end

    r[bounds.hi].temp.v_mul = r[bounds.hi].temp.v_add
    for i = [int64](bounds.hi) - 1, [int64](bounds.lo) - 1, -1 do
      r[i].temp.v_mul = r[i + 1].temp.v_mul * r[i].temp.v_add
    end

    -- TODO: Put these tests back once this pull request is merged and
    --       Regent picks up this change:
    --         https://github.com/zdevito/terra/pull/302
    --
    -- r[bounds.lo].temp.v_min = r[bounds.lo].temp.v_mul
    -- for i = [int64](bounds.lo) + 1, [int64](bounds.hi) + 1 do
    --   r[i].temp.v_min = min(r[i - 1].temp.v_min, r[i].temp.v_mul)
    -- end

    -- r[bounds.hi].temp.v_max = r[bounds.hi].temp.v_min
    -- for i = [int64](bounds.hi) - 1, [int64](bounds.lo) - 1, -1 do
    --   r[i].temp.v_max = max(r[i + 1].temp.v_max, r[i].temp.v_min)
    -- end

    for e in r do
      [(function()
        if DEBUG then
          return rquote
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. "\n"], e.temp.v_add, e.output_cpu.v_add)
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. "\n"], e.temp.v_mul, e.output_cpu.v_mul)
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. "\n"], e.temp.v_min, e.output_cpu.v_min)
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. "\n"], e.temp.v_max, e.output_cpu.v_max)
          end
        else
          return rquote end
        end
      end)()];
      regentlib.assert(e.temp.v_add == e.output_cpu.v_add, "test failed")
      regentlib.assert(e.temp.v_mul == e.output_cpu.v_mul, "test failed")
      regentlib.assert(e.temp.v_min == e.output_cpu.v_min, "test failed")
      regentlib.assert(e.temp.v_max == e.output_cpu.v_max, "test failed");

      [(function()
        if DEBUG then
          return rquote
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. "\n"], e.temp.v_add, e.output_gpu.v_add)
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. "\n"], e.temp.v_mul, e.output_gpu.v_mul)
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. "\n"], e.temp.v_min, e.output_gpu.v_min)
            regentlib.c.printf([format_strings[ty] .. ", " .. format_strings[ty] .. "\n"], e.temp.v_max, e.output_gpu.v_max)
          end
        else
          return rquote end
        end
      end)()];
      regentlib.assert(e.temp.v_add == e.output_gpu.v_add, "test failed")
      regentlib.assert(e.temp.v_mul == e.output_gpu.v_mul, "test failed")
      regentlib.assert(e.temp.v_min == e.output_gpu.v_min, "test failed")
      regentlib.assert(e.temp.v_max == e.output_gpu.v_max, "test failed")
    end
  end

  return init_task, cpu_prefix_task, gpu_prefix_task, check_task
end

local init_tasks = {}
local cpu_prefix_tasks = {}
local gpu_prefix_tasks = {}
local check_tasks = {}

for i = 1, #types do
  init_tasks[types[i]], cpu_prefix_tasks[types[i]], gpu_prefix_tasks[types[i]], check_tasks[types[i]]  = make_tasks(types[i])
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
    local cpu_prefix = cpu_prefix_tasks[ty]
    local gpu_prefix = gpu_prefix_tasks[ty]
    local check = check_tasks[ty]
    return rquote
      var [sym] = region(ispace(int1d, size), fs)
      init([sym])
      cpu_prefix([sym])
      gpu_prefix([sym])
      check([sym])
    end
  end)]
end

task main()
  test(16)
end

regentlib.start(main)
