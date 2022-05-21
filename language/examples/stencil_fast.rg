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

-- runs-with:
-- [["-ll:cpu", "4", "-ntx", "2", "-nty", "2", "-dm:memoize", "-tsteps", "2", "-tprune", "2", "-foverride-demand-cuda", "1"],
--  ["-ll:cpu", "4", "-ntx", "2", "-nty", "2", "-dm:memoize", "-tsteps", "2", "-tprune", "2", "-ffuture", "0", "-foverride-demand-cuda", "1"],
--  ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "4", "-ftrace", "0", "-foverride-demand-cuda", "1"],
--  ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "4", "-tsteps", "2", "-tprune", "2", "-dm:memoize", "-foverride-demand-cuda", "1"],
--  ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "8", "-map_locally", "-ftrace", "0", "-foverride-demand-cuda", "1"]]

-- Inspired by https://github.com/ParRes/Kernels/tree/master/LEGION/Stencil

import "regent"

local format = require("std/format")

local common = require("stencil_common")

local gpuhelper = require("regent/gpu/helper")
local default_foreign = gpuhelper.check_gpu_available() and '0' or '1'

local DTYPE = double
local RADIUS = 2
local USE_FOREIGN = (os.getenv('USE_FOREIGN') or default_foreign) == '1'

local use_python_main = rawget(_G, "stencil_use_python_main") == true

local c = regentlib.c

-- Compile and link stencil.cc
if USE_FOREIGN then
  local root_dir = arg[0]:match(".*/") or "./"
  local stencil_cc = root_dir .. "stencil.cc"
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    stencil_so = out_dir .. "libstencil.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    stencil_so = root_dir .. "libstencil.so"
  else
    stencil_so = os.tmpname() .. ".so" -- root_dir .. "stencil.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local march = os.getenv('MARCH') or 'native'
  local march_flag = '-march=' .. march
  if os.execute("bash -c \"[ `uname` == 'Linux' ]\"") == 0 then
    if os.execute("grep altivec /proc/cpuinfo > /dev/null") == 0 then
      march_flag = '-mcpu=' .. march .. ' -maltivec -mabi=altivec -mvsx'
    end
  end

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  cxx_flags = cxx_flags .. " -O3 " .. march_flag .. " -Wall -Werror -DDTYPE=" .. tostring(DTYPE) .. " -DRESTRICT=__restrict__ -DRADIUS=" .. tostring(RADIUS)
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " " .. stencil_cc .. " -o " .. stencil_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. stencil_cc)
    assert(false)
  end
  regentlib.linklibrary(stencil_so)
  cstencil = terralib.includec("stencil.h", {"-I", root_dir,
                              "-DDTYPE=" .. tostring(DTYPE),
                              "-DRESTRICT=__restrict__",
                              "-DRADIUS=" .. tostring(RADIUS)})
end

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

  local include_path = ""
  local include_dirs = terralib.newlist()
  include_dirs:insert("-I")
  include_dirs:insert(root_dir)
  for path in string.gmatch(os.getenv("INCLUDE_PATH"), "[^;]+") do
    include_path = include_path .. " -I " .. path
    include_dirs:insert("-I")
    include_dirs:insert(path)
  end

  local mapper_cc = root_dir .. "stencil_mapper.cc"
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    mapper_so = out_dir .. "libstencil_mapper.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    mapper_so = root_dir .. "libstencil_mapper.so"
  else
    mapper_so = os.tmpname() .. ".so" -- root_dir .. "stencil_mapper.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  if map_locally then cxx_flags = cxx_flags .. " -DMAP_LOCALLY " end
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                 mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. mapper_cc)
    assert(false)
  end
  regentlib.linklibrary(mapper_so)
  cmapper = terralib.includec("stencil_mapper.h", include_dirs)
end

local min = regentlib.fmin
local max = regentlib.fmax

fspace point {
  input : DTYPE,
  output : DTYPE,
}

fspace timestamp {
  init_start : int64,
  init_stop : int64,
  start : int64,
  stop : int64,
}

if not use_python_main then

terra to_rect(lo : int2d, hi : int2d) : c.legion_rect_2d_t
  return c.legion_rect_2d_t {
    lo = lo:to_point(),
    hi = hi:to_point(),
  }
end

task make_private_partition(points : region(ispace(int2d), point),
                            tiles : ispace(int1d),
                            n : int2d, nt : int2d, radius : int64)
  var coloring = c.legion_domain_point_coloring_create()
  var npoints = n + nt*{ 2*radius, 2*radius }
  for ix = 0, nt.x do
    for iy = 0, nt.y do
      var i = int1d(iy*nt.x + ix)
      var lo = int2d { x = ix*npoints.x/nt.x, y = iy*npoints.y/nt.y }
      var hi = int2d { x = (ix+1)*npoints.x/nt.x - 1, y = (iy+1)*npoints.y/nt.y - 1 }
      var rect = to_rect(lo, hi)
      c.legion_domain_point_coloring_color_domain(
        coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
    end
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_interior_partition(points : region(ispace(int2d), point),
                             tiles : ispace(int1d),
                             n : int2d, nt : int2d, radius : int64)
  var coloring = c.legion_domain_point_coloring_create()
  var npoints = n + nt*{ 2*radius, 2*radius }
  for ix = 0, nt.x do
    for iy = 0, nt.y do
      var i = int1d(iy*nt.x + ix)
      var lo = int2d { x = ix*npoints.x/nt.x + radius, y = iy*npoints.y/nt.y + radius }
      var hi = int2d { x = (ix+1)*npoints.x/nt.x - 1 - radius, y = (iy+1)*npoints.y/nt.y - 1 - radius }
      var rect = to_rect(lo, hi)
      c.legion_domain_point_coloring_color_domain(
        coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
    end
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_exterior_partition(points : region(ispace(int2d), point),
                             tiles : ispace(int1d),
                             n : int2d, nt : int2d, radius : int64)
  var coloring = c.legion_domain_point_coloring_create()
  var npoints = n + nt*{ 2*radius, 2*radius }
  for ix = 0, nt.x do
    for iy = 0, nt.y do
      var i = int1d(iy*nt.x + ix)

      var loffx, loffy = radius, radius
      if ix == 0 then loffx = 0 end
      if iy == 0 then loffy = 0 end

      var hoffx, hoffy = radius, radius
      if ix == nt.x - 1 then hoffx = 0 end
      if iy == nt.y - 1 then hoffy = 0 end

      var lo = int2d { x = ix*npoints.x/nt.x + loffx, y = iy*npoints.y/nt.y + loffy }
      var hi = int2d { x = (ix+1)*npoints.x/nt.x - 1 - hoffx, y = (iy+1)*npoints.y/nt.y - 1 - hoffy }
      var rect = to_rect(lo, hi)
      c.legion_domain_point_coloring_color_domain(
        coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
    end
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

terra clamp(val : int64, lo : int64, hi : int64)
  return min(max(val, lo), hi)
end


function make_ghost_x_partition(is_complete)
  local task ghost_x_partition(points : region(ispace(int2d), point),
                               tiles : ispace(int1d),
                               n : int2d, nt : int2d, radius : int64,
                               dir : int64)
    var coloring = c.legion_domain_point_coloring_create()
    for ix = 0, nt.x do
      for iy = 0, nt.y do
        var i = int1d(iy*nt.x + ix)

        var lo = int2d { x = clamp((ix+dir)*radius, 0, nt.x*radius), y = iy*n.y/nt.y }
        var hi = int2d { x = clamp((ix+1+dir)*radius - 1, -1, nt.x*radius - 1), y = (iy+1)*n.y/nt.y - 1 }
        var rect = to_rect(lo, hi)
        c.legion_domain_point_coloring_color_domain(
          coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
      end
    end
    var p = [(
        function()
          if is_complete then
            return rexpr partition(disjoint, points, coloring, tiles) end
          else
            -- Hack: Since the compiler does not track completeness as
            -- a static property, mark incomplete partitions as aliased.
            return rexpr partition(aliased, points, coloring, tiles) end
          end
        end)()]
    c.legion_domain_point_coloring_destroy(coloring)
    return p
  end
  return ghost_x_partition
end

function make_ghost_y_partition(is_complete)
  local task ghost_y_partition(points : region(ispace(int2d), point),
                               tiles : ispace(int1d),
                               n : int2d, nt : int2d, radius : int64,
                               dir : int64)
    var coloring = c.legion_domain_point_coloring_create()
    for ix = 0, nt.x do
      for iy = 0, nt.y do
        var i = int1d(iy*nt.x + ix)

        var lo = int2d { x = ix*n.x/nt.x, y = clamp((iy+dir)*radius, 0, nt.y*radius) }
        var hi = int2d { x = (ix+1)*n.x/nt.x - 1, y = clamp((iy+1+dir)*radius - 1, -1, nt.y*radius - 1) }
        var rect = to_rect(lo, hi)
        c.legion_domain_point_coloring_color_domain(
          coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
      end
    end
    var p = [(
        function()
          if is_complete then
            return rexpr partition(disjoint, points, coloring, tiles) end
          else
            -- Hack: Since the compiler does not track completeness as
            -- a static property, mark incomplete partitions as aliased.
            return rexpr partition(aliased, points, coloring, tiles) end
          end
        end)()]
    c.legion_domain_point_coloring_destroy(coloring)
    return p
  end
  return ghost_y_partition
end

end -- not use_python_main

local function off(i, x, y)
  return rexpr int2d { x = i.x + x, y = i.y + y } end
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

__demand(__inline)
task get_rect(is : ispace(int2d))
  return c.legion_domain_get_rect_2d(
    c.legion_index_space_get_domain(__runtime(), __raw(is)))
end

terra get_base_and_stride(rect : c.legion_rect_2d_t,
                          physical : c.legion_physical_region_t,
                          field : c.legion_field_id_t)
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  var accessor = c.legion_physical_region_get_field_accessor_array_2d(physical, field)
  var base_pointer = [&DTYPE](c.legion_accessor_array_2d_raw_rect_ptr(
                                      accessor, rect, &subrect, &(offsets[0])))
  regentlib.assert(base_pointer ~= nil, "base pointer is nil")
  regentlib.assert(subrect.lo.x[0] == rect.lo.x[0] and subrect.lo.x[1] == rect.lo.x[1], "subrect not equal to rect")
  regentlib.assert(subrect.hi.x[0] == rect.hi.x[0] and subrect.hi.x[1] == rect.hi.x[1], "subrect not equal to rect")
  regentlib.assert(offsets[0].offset == terralib.sizeof(DTYPE), "stride does not match expected value")

  c.legion_accessor_array_2d_destroy(accessor)

  return { base = base_pointer, stride = offsets[1].offset }
end

local function make_stencil_interior(private, interior, radius)
  if not USE_FOREIGN then
    return rquote
      for i in interior do
        private[i].output = private[i].output +
          [make_stencil_pattern(private, i,  0, -1, radius)] +
          [make_stencil_pattern(private, i, -1,  0, radius)] +
          [make_stencil_pattern(private, i,  1,  0, radius)] +
          [make_stencil_pattern(private, i,  0,  1, radius)]
      end
    end
  else
    return rquote
      var rect = get_rect(private.ispace)
      var { base_input = base, stride_input = stride } = get_base_and_stride(rect, __physical(private)[0], __fields(private)[0])
      var { base_output = base, stride_output = stride } = get_base_and_stride(rect, __physical(private)[1], __fields(private)[1])
      regentlib.assert(stride_output == stride_input, "strides do not match")

      var interior_rect = get_rect(interior.ispace)
      cstencil.stencil(base_input, base_output,
                       stride_input / [terralib.sizeof(DTYPE)],
                       interior_rect.lo.x[0] - rect.lo.x[0], interior_rect.hi.x[0] - rect.lo.x[0] + 1,
                       interior_rect.lo.x[1] - rect.lo.x[1], interior_rect.hi.x[1] - rect.lo.x[1] + 1)
    end
  end
end

local function make_stencil(radius)
  local __demand(__cuda)
        task stencil(private : region(ispace(int2d), point),
                     interior : region(ispace(int2d), point),
                     xm : region(ispace(int2d), point),
                     xp : region(ispace(int2d), point),
                     ym : region(ispace(int2d), point),
                     yp : region(ispace(int2d), point),
                     times : region(ispace(int1d), timestamp),
                     print_ts : bool)
  where
    reads writes(private.{input, output}, times),
    reads(xm.input, xp.input, ym.input, yp.input)
  do
    if print_ts then
      var t = c.legion_get_current_time_in_micros()
      for x in times do x.start = t end
    end

    var interior_rect = get_rect(interior.ispace)
    var interior_lo = int2d { x = interior_rect.lo.x[0], y = interior_rect.lo.x[1] }
    var interior_hi = int2d { x = interior_rect.hi.x[0], y = interior_rect.hi.x[1] }
    var xm_rect = get_rect(xm.ispace)
    var xm_lo = int2d { x = xm_rect.lo.x[0], y = xm_rect.lo.x[1] }
    var xp_rect = get_rect(xp.ispace)
    var xp_lo = int2d { x = xp_rect.lo.x[0], y = xp_rect.lo.x[1] }
    var ym_rect = get_rect(ym.ispace)
    var ym_lo = int2d { x = ym_rect.lo.x[0], y = ym_rect.lo.x[1] }
    var yp_rect = get_rect(yp.ispace)
    var yp_lo = int2d { x = yp_rect.lo.x[0], y = yp_rect.lo.x[1] }

    for i in xm do
      var i2 = i - xm_lo + interior_lo + { -radius, 0 }
      private[i2].input = xm[i].input
    end
    for i in ym do
      var i2 = i - ym_lo + interior_lo + { 0, -radius }
      private[i2].input = ym[i].input
    end
    for i in xp do
      var i2 = i - xp_lo + { x = interior_hi.x + 1, y = interior_lo.y }
      private[i2].input = xp[i].input
    end
    for i in yp do
      var i2 = i - yp_lo + { x = interior_lo.x, y = interior_hi.y + 1 }
      private[i2].input = yp[i].input
    end

    [make_stencil_interior(private, interior, radius)]
  end
  return stencil
end
local stencil = make_stencil(RADIUS)

local function make_increment_interior(private, exterior)
  if not USE_FOREIGN then
    return rquote
      for i in exterior do
        private[i].input = private[i].input + 1
      end
    end
  else
    return rquote
      var rect = get_rect(private.ispace)
      var { base_input = base, stride_input = stride } =
        get_base_and_stride(rect, __physical(private.input)[0], __fields(private.input)[0])

      var exterior_rect = get_rect(exterior.ispace)
      cstencil.increment(base_input,
                         stride_input / [terralib.sizeof(DTYPE)],
                         exterior_rect.lo.x[0] - rect.lo.x[0], exterior_rect.hi.x[0] - rect.lo.x[0] + 1,
                         exterior_rect.lo.x[1] - rect.lo.x[1], exterior_rect.hi.x[1] - rect.lo.x[1] + 1)
    end
  end
end

__demand(__cuda)
task increment(private : region(ispace(int2d), point),
               exterior : region(ispace(int2d), point),
               xm : region(ispace(int2d), point),
               xp : region(ispace(int2d), point),
               ym : region(ispace(int2d), point),
               yp : region(ispace(int2d), point),
               times : region(ispace(int1d), timestamp),
               init_ts : bool,
               print_ts : bool)
where reads writes(private.input, xm.input, xp.input, ym.input, yp.input, times) do
  [make_increment_interior(private, exterior)]
  for i in xm do i.input += 1 end
  for i in xp do i.input += 1 end
  for i in ym do i.input += 1 end
  for i in yp do i.input += 1 end

  var t = c.legion_get_current_time_in_micros()
  if init_ts then
    for x in times do x.init_stop = t end
  end
  if print_ts then
    for x in times do x.stop = t end
  end
end

task check(private : region(ispace(int2d), point),
           interior : region(ispace(int2d), point),
           tsteps : int64, init : int64)
where reads(private.{input, output}) do
  var expect_in = init + tsteps
  var expect_out = init
  var num_input_failed : int64 = 0
  for i in interior do
    if private[i].input ~= expect_in then
      if num_input_failed == 0 then
        format.println("input ({},{}): {.3} should be {}",
                       i.x, i.y, private[i].input, expect_in)
      end
      num_input_failed += 1
    end
  end
  var num_output_failed : int64 = 0
  for i in interior do
    if private[i].output ~= expect_out then
      if num_output_failed == 0 then
        format.println("output ({},{}): {.3} should be {}",
                       i.x, i.y, private[i].output, expect_out)
      end
      num_output_failed += 1
    end
  end
  if num_input_failed > 0 then
    format.println("total number of bad input entries: {}",
                   num_input_failed)
  end
  if num_output_failed > 0 then
    format.println("total number of bad output entries: {}",
                   num_output_failed)
  end
  regentlib.assert(num_input_failed == 0, "test failed: input mismatch")
  regentlib.assert(num_output_failed == 0, "test failed: output mismatch")
  format.println("check completed successfully")
end

task fill_(r : region(ispace(int2d), point), v : DTYPE)
where reads writes(r.{input, output}) do
  -- fill(r.{input, output}, v)
  for x in r do x.input = v end
  for x in r do x.output = v end
end

if not use_python_main then

task read_config()
  return common.read_config()
end

task begin_init(times : region(ispace(int1d), timestamp))
where writes(times) do
  var t = c.legion_get_current_time_in_micros()
  for x in times do x.init_start = t end
end

task get_elapsed(all_times : region(ispace(int1d), timestamp))
where reads(all_times) do
  var init_start = [int64:max()]
  var init_stop = [int64:min()]
  var start = [int64:max()]
  var stop = [int64:min()]

  for t in all_times do
    init_start min= t.init_start
    init_stop max= t.init_stop
    start min= t.start
    stop max= t.stop
  end

  return { init_time = 1e-6 * (init_stop - init_start), sim_time = 1e-6 * (stop - start) }
end

task print_time(color : int, init_time : double, sim_time : double)
  if color == 0 then
    format.println("INIT TIME = {7.3} s", init_time)
    format.println("ELAPSED TIME = {7.3} s", sim_time)
  end
end

__demand(__inner, __replicable)
task main()
  var conf = read_config()

  var nbloated = int2d { conf.nx, conf.ny } -- Grid size along each dimension, including border.
  var nt = int2d { conf.ntx, conf.nty } -- Number of tiles to make in each dimension.
  var init : int64 = conf.init

  var radius : int64 = RADIUS
  var n = nbloated - { 2*radius, 2*radius } -- Grid size, minus the border.
  regentlib.assert(n >= nt, "grid too small")

  var grid = ispace(int2d, n + nt*{ 2*radius, 2*radius })
  var nt2 = nt.x*nt.y
  var tiles = ispace(int1d, nt2)

  var times = region(ispace(int1d, nt2), timestamp)
  var p_times = partition(equal, times, ispace(int1d, nt2))
  fill(times.{init_start, init_stop, start, stop}, 0)

  __fence(__execution, __block)
  __demand(__index_launch)
  for i in tiles do
    begin_init(p_times[i])
  end
  __fence(__execution, __block)

  var points = region(grid, point)
  var private = make_private_partition(points, tiles, n, nt, radius)
  var interior = make_interior_partition(points, tiles, n, nt, radius)
  var exterior = make_exterior_partition(points, tiles, n, nt, radius)

  var xm = region(ispace(int2d, { x = nt.x*radius, y = n.y }), point)
  var xp = region(ispace(int2d, { x = nt.x*radius, y = n.y }), point)
  var ym = region(ispace(int2d, { x = n.x, y = nt.y*radius }), point)
  var yp = region(ispace(int2d, { x = n.x, y = nt.y*radius }), point)
  var pxm_in = [make_ghost_x_partition(false)](xm, tiles, n, nt, radius, -1)
  var pxp_in = [make_ghost_x_partition(false)](xp, tiles, n, nt, radius,  1)
  var pym_in = [make_ghost_y_partition(false)](ym, tiles, n, nt, radius, -1)
  var pyp_in = [make_ghost_y_partition(false)](yp, tiles, n, nt, radius,  1)
  var pxm_out = [make_ghost_x_partition(true)](xm, tiles, n, nt, radius, 0)
  var pxp_out = [make_ghost_x_partition(true)](xp, tiles, n, nt, radius, 0)
  var pym_out = [make_ghost_y_partition(true)](ym, tiles, n, nt, radius, 0)
  var pyp_out = [make_ghost_y_partition(true)](yp, tiles, n, nt, radius, 0)

  fill(points.{input, output}, init)
  fill(xm.{input, output}, init)
  fill(xp.{input, output}, init)
  fill(ym.{input, output}, init)
  fill(yp.{input, output}, init)

  var tprune : int64 = conf.tprune
  var tsteps : int64 = conf.tsteps + 2 * tprune

  -- __demand(__spmd)
  -- do
  --   for i = 0, nt2 do
  --     fill_(pxm_out[i], init)
  --   end
  --   for i = 0, nt2 do
  --     fill_(pxp_out[i], init)
  --   end
  --   for i = 0, nt2 do
  --     fill_(pym_out[i], init)
  --   end
  --   for i = 0, nt2 do
  --     fill_(pyp_out[i], init)
  --   end
  -- end

  __fence(__execution, __block)

  __demand(__spmd)
  do
    -- for i = 0, nt2 do
    --   fill_(private[i], init)
    -- end

    __demand(__trace)
    for t = 0, tsteps do
      -- __demand(__index_launch)
      for i in tiles do
        stencil(private[i], interior[i], pxm_in[i], pxp_in[i], pym_in[i], pyp_in[i], p_times[i], t == tprune)
      end
      -- __demand(__index_launch)
      for i in tiles do
        increment(private[i], exterior[i], pxm_out[i], pxp_out[i], pym_out[i], pyp_out[i], p_times[i], t == 0, t == tsteps - tprune - 1)
      end
    end

    for i in tiles do
      check(private[i], interior[i], tsteps, init)
    end
  end

  var { init_time, sim_time } = get_elapsed(times)
  for i = 0, nt2 do print_time(i, init_time, sim_time) end
end

else -- not use_python_main

extern task main()
main:set_task_id(2)

end -- not use_python_main

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
  local link_flags = terralib.newlist({"-L" .. out_dir, "-lstencil_mapper"})
  if USE_FOREIGN then
    link_flags:insert("-lstencil")
  end

  if os.getenv('STANDALONE') == '1' then
    os.execute('cp ' .. os.getenv('LG_RT_DIR') .. '/../bindings/regent/' ..
        regentlib.binding_library .. ' ' .. out_dir)
  end

  local exe = os.getenv('OBJNAME') or "stencil"
  regentlib.saveobj(main, exe, "executable", cmapper.register_mappers, link_flags)
else
  regentlib.start(main, cmapper.register_mappers)
end
