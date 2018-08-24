-- Copyright 2018 Stanford University, Los Alamos National Laboratory
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

function make_fs(ty)
  local fspace wrap
  {
    v : ty;
  }
  local fspace fs
  {
    input : wrap;
    output : wrap;
    temp : wrap;
  }
  return fs
end

local types = terralib.newlist { double, float, uint32, int64 }
local formats = terralib.newlist { "%lf", "%f", "%u", "%ld" }

local fs_types = {}
local format_strings = {}
for i = 1, #types do
  fs_types[types[i]] = make_fs(types[i])
  format_strings[types[i]] = formats[i]
end

function make_tasks(ty)
  local init_task, prefix_task, check_task
  local fs_type = fs_types[ty]

  task init_task(r : region(ispace(int1d), fs_type))
  where
    reads writes(r)
  do
    for e in r do
      r[e].input.v = ([ty](e) + 5) % 10 + 1
      r[e].output.v = [ty](0)
    end
  end

  task prefix_task(r : region(ispace(int1d), fs_type))
  where
    reads(r.input), reads writes(r.output)
  do
    var dir1 : int1d = [int1d](1)
    var dir2 : int1d = [int1d](-1)
    __parallel_prefix(r.output.v, r.input.v,  +, dir1)
    __parallel_prefix(r.output.v, r.output.v, min, dir2)
  end

  task check_task(r : region(ispace(int1d), fs_type))
  where
    reads(r.input), reads(r.output), reads writes(r.temp)
  do
    var bounds = r.bounds
    r[bounds.lo].temp.v = r[bounds.lo].input.v
    for i = [int64](bounds.lo) + 1, [int64](bounds.hi) + 1 do
      r[i].temp.v = r[i - 1].temp.v + r[i].input.v
    end
    for i = [int64](bounds.hi) - 1, [int64](bounds.lo) - 1, -1 do
      r[i].temp.v = min(r[i + 1].temp.v, r[i].temp.v)
    end
    for e in r do
      regentlib.assert(e.temp.v == e.output.v, "test failed")
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

task main()
  [types:map(function(ty)
    local fs = fs_types[ty]
    local sym = region_symbols[ty]
    local init = init_tasks[ty]
    local prefix = prefix_tasks[ty]
    local check = check_tasks[ty]
    return rquote
      var [sym] = region(ispace(int1d, 10), fs)
      init([sym])
      prefix([sym])
      check([sym])
    end
  end)]
end

regentlib.start(main)
