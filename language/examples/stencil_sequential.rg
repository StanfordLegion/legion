-- Copyright 2019 Stanford University
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
-- [["-ll:cpu", "4", "-ntx", "2", "-nty", "2", "-dm:memoize", "-tsteps", "2", "-tprune", "2"]]

--,
--  ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "4", "-ftrace", "0"],
--  ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "4", "-tsteps", "2", "-tprune", "2", "-dm:memoize"],
--  ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "8", "-map_locally", "-ftrace", "0"]]

-- Inspired by https://github.com/ParRes/Kernels/tree/master/LEGION/Stencil

import "regent"

local common = require("stencil_common")

local DTYPE = double
local RADIUS = 2

local c = regentlib.c

local map_locally = false
do
  local cstring = terralib.includec("string.h")
  for _, arg in ipairs(arg) do
    if cstring.strcmp(arg, "-map_locally") == 0 then
      map_locally = true
      break
    end
  end
end

do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
  local mapper_cc = root_dir .. "stencil_sequential_mapper.cc"
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    mapper_so = out_dir .. "libstencil_sequential_mapper.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    mapper_so = root_dir .. "libstencil_sequential_mapper.so"
  else
    mapper_so = os.tmpname() .. ".so" -- root_dir .. "stencil_mapper.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CC_FLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  if map_locally then cxx_flags = cxx_flags .. " -DMAP_LOCALLY " end
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. mapper_cc)
    assert(false)
  end
  terralib.linklibrary(mapper_so)
  cmapper = terralib.includec("stencil_mapper.h", {"-I", root_dir, "-I", runtime_dir})
end

fspace point {
  input : DTYPE,
  output : DTYPE,
}

local function off(i, x, y)
  return rexpr i + { x, y } end
end

local function make_stencil_pattern(points, index, off_x, off_y, radius)
  local value
  for i = 1, radius do
    local neg = off_x < 0 or off_y < 0
    local coeff = ((neg and -1) or 1)/(2*i*radius)
    local x, y = off_x*i, off_y*i
    local component = rexpr coeff*points[ [off(index, x, y)] ].input end
    if value then
      value = rexpr value + component end
    else
      value = rexpr component end
    end
  end
  return value
end

__demand(__parallel, __cuda)
task stencil(points   : region(ispace(int2d), point),
             interior : region(ispace(int2d), point))
where
  reads writes(points.{input, output}),
  interior <= points
do
  for i in interior do
    points[i].output = points[i].output +
      [make_stencil_pattern(points, i,  0, -1, RADIUS)] +
      [make_stencil_pattern(points, i, -1,  0, RADIUS)] +
      [make_stencil_pattern(points, i,  1,  0, RADIUS)] +
      [make_stencil_pattern(points, i,  0,  1, RADIUS)]
  end
end

__demand(__parallel, __cuda)
task increment(points : region(ispace(int2d), point))
where reads writes(points.input) do
  for i in points do
    points[i].input = points[i].input + 1
  end
end

__demand(__parallel)
task check(points   : region(ispace(int2d), point),
           interior : region(ispace(int2d), point),
           tsteps : int64, init : int64)
where
  reads(points.{input, output}),
  interior <= points
do
  var expect_in = init + tsteps
  var expect_out = init
  for i in interior do
    regentlib.assert(points[i].input == expect_in, "test failed")
    regentlib.assert(points[i].output == expect_out, "test failed")
  end
end

task make_interior_partition(points : region(ispace(int2d), point),
                             n : int2d,
                             radius : int64)
  var r = region(ispace(int1d, 1), rect2d)
  r[0] = rect2d { lo = {radius, radius}, hi = n + {radius - 1, radius - 1} }
  var p = partition(equal, r, ispace(int1d, 1))
  return image(points, p, r)
end

__demand(__inner)
task main()
  var conf = common.read_config()

  var nbloated = int2d { conf.nx, conf.ny } -- Grid size along each dimension, including border.
  var nt = int2d { conf.ntx, conf.nty } -- Number of tiles to make in each dimension.
  var init : int64 = conf.init

  var radius : int64 = RADIUS
  var n = nbloated - { 2*radius, 2*radius } -- Grid size, minus the border.
  regentlib.assert(n >= nt, "grid too small")

  var grid = ispace(int2d, nbloated)
  var tiles = ispace(int2d, nt)

  var points = region(grid, point)
  var p_interior = make_interior_partition(points, n, radius)
  var interior = p_interior[0]

  fill(points.{input, output}, init)

  var tprune : int64 = conf.tprune
  var tsteps : int64 = conf.tsteps + 2 * tprune

  --__demand(__spmd)
  __parallelize_with tiles
  do
    __fence(__execution, __block)
    var ts_start = c.legion_get_current_time_in_micros()
    var ts_end = ts_start
    __demand(__spmd)
    for t = 0, tsteps do
      if t == tprune then
        __fence(__execution, __block)
        ts_start = c.legion_get_current_time_in_micros()
      end

      stencil(points, interior)
      increment(points)

      if t == tsteps - tprune - 1 then
        __fence(__execution, __block)
        ts_end = c.legion_get_current_time_in_micros()
      end
    end
    var sim_time = 1e-6 * (ts_end - ts_start)
    c.printf("ELAPSED TIME = %7.3f s\n", sim_time)
    check(points, interior, tsteps, init)
  end
end

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
  local link_flags = terralib.newlist({"-L" .. out_dir, "-lstencil_sequential_mapper"})

  if os.getenv('STANDALONE') == '1' then
    os.execute('cp ' .. os.getenv('LG_RT_DIR') .. '/../bindings/regent/libregent.so ' .. out_dir)
  end

  local exe = os.getenv('OBJNAME') or "stencil_sequential"
  regentlib.saveobj(main, exe, "executable", cmapper.register_mappers, link_flags)
else
  regentlib.start(main, cmapper.register_mappers)
end
