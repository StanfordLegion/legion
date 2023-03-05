-- Copyright 2023 Stanford University, NVIDIA Corporation
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
-- [["-ll:gpu", "1" ]]

import "regent"

local format = require("std/format")

local gpu_inc_task = terralib.memoize(function(index_type, field_type)
  local __demand(__cuda)
  task t(r : region(ispace(index_type), field_type), x : field_type)
  where reads writes(r) do
    for e in r do
      r[e] += x -- just do something so we know the task is running
    end
  end
  t:set_name("gpu_inc_" .. tostring(index_type) .. "_" .. tostring(field_type))
  return t
end)

local check_task = terralib.memoize(function(index_type, field_type)
  local task t(r : region(ispace(index_type), field_type), c : field_type)
  where reads writes(r) do
    for e in r do
      if r[e] ~= c then
        format.println("dummy expected 0x{x} but got: r[{}] = 0x{x}", c, e, r[e])
        regentlib.assert(false, "test failed")
      end
    end
  end
  t:set_name("check_" .. tostring(index_type) .. "_" .. tostring(field_type))
  return t
end)

task write_partitions_1d(s : region(ispace(int1d), rect1d))
where reads writes(s) do
  s[0] = rect1d { int1d(0), int1d(1) }
  s[1] = rect1d { int1d(4), int1d(5) }
  s[2] = rect1d { int1d(2), int1d(3) }
  s[3] = rect1d { int1d(6), int1d(7) }
end

task write_partitions_2d(s : region(ispace(int1d), rect2d))
where reads writes(s) do
  s[0] = rect2d { int2d { 0, 0 }, int2d { 1, 1 } }
  s[1] = rect2d { int2d { 2, 2 }, int2d { 3, 3 } }
  s[2] = rect2d { int2d { 4, 4 }, int2d { 5, 5 } }
  s[3] = rect2d { int2d { 6, 6 }, int2d { 7, 7 } }
end

task write_partitions_3d(s : region(ispace(int1d), rect3d))
where reads writes(s) do
  s[0] = rect3d { int3d { 0, 0, 0 }, int3d { 1, 1, 1 } }
  s[1] = rect3d { int3d { 2, 2, 2 }, int3d { 3, 3, 3 } }
  s[2] = rect3d { int3d { 4, 4, 4 }, int3d { 5, 5, 5 } }
  s[3] = rect3d { int3d { 6, 6, 6 }, int3d { 7, 7, 7 } }
end

local get_write_partitions = {
  [int1d] = write_partitions_1d,
  [int2d] = write_partitions_2d,
  [int3d] = write_partitions_3d,
}

local get_bounds_initializer = {
  [int1d] = rexpr 8 end,
  [int2d] = rexpr {8, 8} end,
  [int3d] = rexpr {8, 8, 8} end,
}

local get_rect_type = {
  [int1d] = rect1d,
  [int2d] = rect2d,
  [int3d] = rect3d,
}

local test_task = terralib.memoize(function(index_type, field_type, init_value, inc_value, check_value)
  local gpu_inc = gpu_inc_task(index_type, field_type)
  local check = check_task(index_type, field_type)
  local rect_type = get_rect_type[index_type]
  local bounds_initializer = get_bounds_initializer[index_type]
  local write_partitions = get_write_partitions[index_type]

  local __demand(__inner)
  task t()
    var r = region(ispace(index_type, [bounds_initializer]), field_type)
    fill(r, init_value)
    var s = region(ispace(int1d, 4), rect_type)
    write_partitions(s)
    var s_p = partition(equal, s, ispace(int1d, 2))
    var r_p = image(disjoint, r, s_p, s)
    __demand(__index_launch)
    for c in r_p.colors do
      gpu_inc(r_p[c], inc_value)
    end
    __demand(__index_launch)
    for c in r_p.colors do
      check(r_p[c], check_value)
    end
  end
  t:set_name("test_" .. tostring(index_type) .. "_" .. tostring(field_type))
  return t
end)

task toplevel()
  [test_task(int1d, int8,  0x12,                 0x44,                 0x56)]();
  [test_task(int1d, int16, 0x1234,               0x4444,               0x5678)]();
  [test_task(int1d, int32, 0x12345678,           0x44444444,           0x56789ABC)]();
  [test_task(int1d, int64, 0x123456789ABCDEF0LL, 0x4444444444444444LL, 0x56789ABCDF012334LL)]();

  [test_task(int2d, int8,  0x12,                 0x44,                 0x56)]();
  [test_task(int2d, int16, 0x1234,               0x4444,               0x5678)]();
  [test_task(int2d, int32, 0x12345678,           0x44444444,           0x56789ABC)]();
  [test_task(int2d, int64, 0x123456789ABCDEF0LL, 0x4444444444444444LL, 0x56789ABCDF012334LL)]();

  [test_task(int3d, int8,  0x12,                 0x44,                 0x56)]();
  [test_task(int3d, int16, 0x1234,               0x4444,               0x5678)]();
  [test_task(int3d, int32, 0x12345678,           0x44444444,           0x56789ABC)]();
  [test_task(int3d, int64, 0x123456789ABCDEF0LL, 0x4444444444444444LL, 0x56789ABCDF012334LL)]();
end

regentlib.start(toplevel)
