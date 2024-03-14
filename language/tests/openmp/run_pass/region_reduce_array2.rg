-- Copyright 2024 Stanford University, NVIDIA Corporation
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
-- [["-fopenmp", "1", "-ll:ocpu", "1", "-ll:othr", "4", "-ll:cpu", "0", "-ll:okindhack" ]]

import "regent"

local function generate_fs(type)
  local fs
  fspace fs
  {
    input : type,
    output_add : type[1],
    output_mul : type[1],
  }
  return fs
end

local function generate_init(type)
  local tsk
  task tsk(r : region(ispace(int1d), type))
  where
    reads writes(r)
  do
    for e in r do
      e.input = 2
      e.output_add[0] = 0
      e.output_mul[0] = 1
   end
  end
  return tsk
end

local function generate_red(type)
  local tsk
  task tsk(is   : ispace(int1d),
           size : int,
           rep  : int,
           r    : region(ispace(int1d), type))
  where
    reads(r.input), reads writes(r.{output_add, output_mul})
  do
    __demand(__openmp)
    for p in is do
      var target = [int](p) % size
      var v = r[target].input
      r[target].output_add[0] += v
      r[target].output_mul[0] *= v
    end
  end
  return tsk
end

local function generate_check(type)
  local tsk
  task tsk(r   : region(ispace(int1d), type),
           rep : int)
  where
    reads(r.{output_add, output_mul})
  do
    var p = 1
    for i = 0, rep do p *= 2 end
    for e in r do
      regentlib.assert(e.output_add[0] == rep * 2, "test failed")
      regentlib.assert(e.output_mul[0] == p, "test failed")
    end
  end
  return tsk
end

local types = terralib.newlist({
  int8, int16, int32, int64,
  uint8, uint16, uint32, uint64,
  float, double
})

local fs_types = types:map(generate_fs)
local init_tasks = fs_types:map(generate_init)
local red_tasks = fs_types:map(generate_red)
local check_tasks = fs_types:map(generate_check)

local test_tasks = terralib.newlist()
local function generate_test(idx)
  local tsk
  local fs = fs_types[idx]
  local init_task = init_tasks[idx]
  local red_task = red_tasks[idx]
  local check_task = check_tasks[idx]
  task tsk(size : int, rep : int)
    var is = ispace(int1d, size * rep)
    var r = region(ispace(int1d, size), fs)

    init_task(r)
    red_task(is, size, rep, r)
    check_task(r, rep)
  end
  return tsk
end
for idx = 1, #types do
  test_tasks:insert(generate_test(idx))
end

task toplevel()
  [test_tasks:map(function(test_task)
    return rquote
      [test_task](100, 5)
    end
  end)]
end

regentlib.start(toplevel)
