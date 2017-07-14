-- Copyright 2015 Stanford University
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
--   ["pennant.tests/sedovsmall/sedovsmall.pnt",
--    "-npieces", "1", "-compact", "0"],
--   ["pennant.tests/sedovsmall/sedovsmall.pnt",
--    "-npieces", "2", "-compact", "0",
--    "-absolute", "1e-6", "-relative", "1e-6", "-relative_absolute", "1e-9"],
--   ["pennant.tests/sedov/sedov.pnt", "-npieces", "1", "-compact", "0",
--    "-absolute", "2e-6", "-relative", "1e-8", "-relative_absolute", "1e-10"],
--   ["pennant.tests/sedov/sedov.pnt", "-npieces", "3", "-compact", "0",
--    "-absolute", "2e-6", "-relative", "1e-8", "-relative_absolute", "1e-10"],
--   ["pennant.tests/leblanc/leblanc.pnt", "-npieces", "1", "-compact", "0"],
--   ["pennant.tests/leblanc/leblanc.pnt", "-npieces", "2", "-compact", "0"]
-- ]

-- Inspired by https://github.com/losalamos/PENNANT

-- (terra() c.printf("Compiling C++ module (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6) end)()

import "regent"

-- Compile and link pennant.cc
local cpennant
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = root_dir .. "../../runtime"
  local legion_dir = root_dir .. "../../runtime/legion"
  local mapper_dir = root_dir .. "../../runtime/mappers"
  local realm_dir = root_dir .. "../../runtime/realm"
  local pennant_cc = root_dir .. "pennant.cc"
  local pennant_so = os.tmpname() .. ".so" -- root_dir .. "pennant.so"
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = "-O2 -std=c++0x -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 " -I " .. mapper_dir .. " " .. " -I " .. legion_dir .. " " ..
                 " -I " .. realm_dir .. " " .. pennant_cc .. " -o " .. pennant_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. pennant_cc)
    assert(false)
  end
  terralib.linklibrary(pennant_so)
  cpennant = terralib.includec("pennant.h", {"-I", root_dir, "-I", runtime_dir,
                                             "-I", mapper_dir, "-I", legion_dir})
end

local c = regentlib.c

-- Include other headers
local cmath = terralib.includec("math.h")
local cstring = terralib.includec("string.h")

local sqrt = terralib.intrinsic("llvm.sqrt.f64", double -> double)

(terra() c.printf("Compiling from top (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6) end)()

-- #####################################
-- ## Data Structures
-- #################

-- Import max/min for Terra
local max = regentlib.fmax
local min = regentlib.fmin

terra abs(a : double) : double
  if a < 0 then
    return -a
  else
    return a
  end
end

struct vec2 {
  x : double,
  y : double,
}

terra vec2.metamethods.__add(a : vec2, b : vec2) : vec2
  return vec2 { x = a.x + b.x, y = a.y + b.y }
end

terra vec2.metamethods.__sub(a : vec2, b : vec2) : vec2
  return vec2 { x = a.x - b.x, y = a.y - b.y }
end

vec2.metamethods.__mul = terralib.overloadedfunction(
  "__mul", {
    terra(a : double, b : vec2) : vec2
      return vec2 { x = a * b.x, y = a * b.y }
    end,
    terra(a : vec2, b : double) : vec2
      return vec2 { x = a.x * b, y = a.y * b }
    end
  })

terra dot(a : vec2, b : vec2) : double
  return a.x*b.x + a.y*b.y
end

terra cross(a : vec2, b : vec2) : double
  return a.x*b.y - a.y*b.x
end

terra length(a : vec2) : double
  return sqrt(dot(a, a))
end

terra rotateCCW(a : vec2) : vec2
  return vec2 { x = -a.y, y = a.x }
end

terra project(a : vec2, b : vec2)
  return a - b*dot(a, b)
end

local config_fields_input = terralib.newlist({
  -- Configuration variables.
  {field = "alfa", type = double, default_value = 0.5},
  {field = "bcx", type = double[2], default_value = `array(0.0, 0.0), linked_field = "bcx_n"},
  {field = "bcx_n", type = int64, default_value = 0, is_linked_field = true},
  {field = "bcy", type = double[2], default_value = `array(0.0, 0.0), linked_field = "bcy_n"},
  {field = "bcy_n", type = int64, default_value = 0, is_linked_field = true},
  {field = "cfl", type = double, default_value = 0.6},
  {field = "cflv", type = double, default_value = 0.1},
  {field = "chunksize", type = int64, default_value = 99999999},
  {field = "cstop", type = int64, default_value = 999999},
  {field = "dtfac", type = double, default_value = 1.2},
  {field = "dtinit", type = double, default_value = 1.0e99},
  {field = "dtmax", type = double, default_value = 1.0e99},
  {field = "dtreport", type = double, default_value = 10},
  {field = "einit", type = double, default_value = 0.0},
  {field = "einitsub", type = double, default_value = 0.0},
  {field = "gamma", type = double, default_value = 5.0 / 3.0},
  {field = "meshscale", type = double, default_value = 1.0},
  {field = "q1", type = double, default_value = 0.0},
  {field = "q2", type = double, default_value = 2.0},
  {field = "qgamma", type = double, default_value = 5.0 / 3.0},
  {field = "rinit", type = double, default_value = 1.0},
  {field = "rinitsub", type = double, default_value = 1.0},
  {field = "ssmin", type = double, default_value = 0.0},
  {field = "subregion", type = double[4], default_value = `arrayof(double, 0, 0, 0, 0), linked_field = "subregion_n"},
  {field = "subregion_n", type = int64, default_value = 0, is_linked_field = true},
  {field = "tstop", type = double, default_value = 1.0e99},
  {field = "uinitradial", type = double, default_value = 0.0},
  {field = "meshparams", type = double[4], default_value = `arrayof(double, 0, 0, 0, 0), linked_field = "meshparams_n"},
  {field = "meshparams_n", type = int64, default_value = 0, is_linked_field = true},
})

local config_fields_meshgen = terralib.newlist({
  -- Mesh generator variables.
  {field = "meshtype", type = int64, default_value = 0},
  {field = "nzx", type = int64, default_value = 0},
  {field = "nzy", type = int64, default_value = 0},
  {field = "numpcx", type = int64, default_value = 0},
  {field = "numpcy", type = int64, default_value = 0},
  {field = "lenx", type = double, default_value = 0.0},
  {field = "leny", type = double, default_value = 0.0},
})

local config_fields_mesh = terralib.newlist({
  -- Mesh variables.
  {field = "nz", type = int64, default_value = 0},
  {field = "np", type = int64, default_value = 0},
  {field = "ns", type = int64, default_value = 0},
  {field = "maxznump", type = int64, default_value = 0},
})

local config_fields_cmd = terralib.newlist({
  -- Command-line parameters.
  {field = "npieces", type = int64, default_value = 1},
  {field = "use_foreign", type = bool, default_value = false},
  {field = "enable", type = bool, default_value = true},
  {field = "warmup", type = bool, default_value = true},
  {field = "compact", type = bool, default_value = true},
  {field = "stripsize", type = int64, default_value = 128},
  {field = "spansize", type = int64, default_value = 512},
  {field = "nocheck", type = bool, default_value = false},
})

local config_fields_all = terralib.newlist()
config_fields_all:insertall(config_fields_input)
config_fields_all:insertall(config_fields_meshgen)
config_fields_all:insertall(config_fields_mesh)
config_fields_all:insertall(config_fields_cmd)

local config = terralib.types.newstruct("config")
config.entries:insertall(config_fields_all)

local MESH_PIE = 0
local MESH_RECT = 1
local MESH_HEX = 2

fspace zone {
  zxp :    vec2,         -- zone center coordinates, middle of cycle
  zx :     vec2,         -- zone center coordinates, end of cycle
  zareap : double,       -- zone area, middle of cycle
  zarea :  double,       -- zone area, end of cycle
  zvol0 :  double,       -- zone volume, start of cycle
  zvolp :  double,       -- zone volume, middle of cycle
  zvol :   double,       -- zone volume, end of cycle
  zdl :    double,       -- zone characteristic length
  zm :     double,       -- zone mass
  zrp :    double,       -- zone density, middle of cycle
  zr :     double,       -- zone density, end of cycle
  ze :     double,       -- zone specific energy
  zetot :  double,       -- zone total energy
  zw :     double,       -- zone work
  zwrate : double,       -- zone work rate
  zp :     double,       -- zone pressure
  zss :    double,       -- zone sound speed
  zdu :    double,       -- zone delta velocity (???)

  -- Temporaries for QCS
  zuc :    vec2,         -- zone center velocity
  z0tmp :  double,       -- temporary for qcs_vel_diff

  -- Placed at end to avoid messing up alignment
  znump :  uint8,        -- number of points in zone
}

fspace point {
  px0 : vec2,            -- point coordinates, start of cycle
  pxp : vec2,            -- point coordinates, middle of cycle
  px :  vec2,            -- point coordinates, end of cycle
  pu0 : vec2,            -- point velocity, start of cycle
  pu :  vec2,            -- point velocity, end of cycle
  pap : vec2,            -- point acceleration, middle of cycle -- FIXME: dead
  pf :  vec2,            -- point force
  pmaswt : double,       -- point mass

  -- Used for computing boundary conditions
  has_bcx : bool,
  has_bcy : bool,
}

fspace side(rz : region(zone),
            rpp : region(point),
            rpg : region(point),
            rs : region(side(rz, rpp, rpg, rs))) {
  mapsz :  ptr(zone, rz),                      -- maps: side -> zone
  mapsp1 : ptr(point, rpp, rpg),               -- maps: side -> points 1 and 2
  mapsp2 : ptr(point, rpp, rpg),
  mapss3 : ptr(side(rz, rpp, rpg, rs), rs),    -- maps: side -> previous side
  mapss4 : ptr(side(rz, rpp, rpg, rs), rs),    -- maps: side -> next side

  sareap : double,       -- side area, middle of cycle
  sarea :  double,       -- side area, end of cycle
  svolp :  double,       -- side volume, middle of cycle -- FIXME: dead field
  svol :   double,       -- side volume, end of cycle    -- FIXME: dead field
  ssurfp : vec2,         -- side surface vector, middle of cycle -- FIXME: dead
  smf :    double,       -- side mass fraction
  sfp :    vec2,         -- side force, pgas
  sft :    vec2,         -- side force, tts
  sfq :    vec2,         -- side force, qcs

  -- In addition to storing their own state, sides also store the
  -- state of edges and corners. This can be done because there is a
  -- 1-1 correspondence between sides and edges/corners. Technically,
  -- edges can be shared between zones, but the computations on edges
  -- are minimal, and are not actually used for sharing information,
  -- so duplicating computations on edges is inexpensive.

  -- Edge variables
  exp :    vec2,         -- edge center coordinates, middle of cycle
  ex :     vec2,         -- edge center coordinates, end of cycle
  elen :   double,       -- edge length, end of cycle

  -- Corner variables (temporaries for QCS)
  carea :  double,       -- corner area
  cevol :  double,       -- corner evol
  cdu :    double,       -- corner delta velocity
  cdiv :   double,       -- ??????????
  ccos :   double,       -- corner cosine
  cqe1 :   vec2,         -- ??????????
  cqe2 :   vec2,         -- ??????????
}

-- #####################################
-- ## Initialization
-- #################

function _t1() end -- seems like I need this to break up the statements
(terra() c.printf("Compiling kernels (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6) end)()

-- Hack: This exists to make the compiler recompute the bitmasks for
-- each pointer. This needs to happen here (rather than at
-- initialization time) because we subverted the type system in the
-- construction of the mesh pieces.
task init_pointers(rz : region(zone), rpp : region(point), rpg : region(point),
                   rs : region(side(rz, rpp, rpg, rs)))
where
  reads writes(rs.{mapsp1, mapsp2})
do
  for s in rs do
    s.mapsp1 = dynamic_cast(ptr(point, rpp, rpg), s.mapsp1)
    regentlib.assert(not isnull(s.mapsp1), "dynamic_cast failed on mapsp1")
    s.mapsp2 = dynamic_cast(ptr(point, rpp, rpg), s.mapsp2)
    regentlib.assert(not isnull(s.mapsp2), "dynamic_cast failed on mapsp2")
  end
  return 1
end

task init_mesh_zones(rz : region(zone))
where
  writes(rz.{zx, zarea, zvol})
do
  for z in rz do
    z.zx = vec2 {x = 0.0, y = 0.0}
    z.zarea = 0.0
    z.zvol = 0.0
  end
end

-- Call calc_centers_full.
-- Call calc_volumes_full.

task init_side_fracs(rz : region(zone), rpp : region(point), rpg : region(point),
                     rs : region(side(rz, rpp, rpg, rs)))
where
  reads(rz.zarea, rs.{mapsz, sarea}),
  writes(rs.smf)
do
  for s in rs do
    var z = s.mapsz

    s.smf = s.sarea / z.zarea
  end
end

task init_hydro(rz : region(zone), rinit : double, einit : double,
                rinitsub : double, einitsub : double,
                subregion_x0 : double, subregion_x1 : double,
                subregion_y0 : double, subregion_y1 : double)
where
  reads(rz.{zx, zvol}),
  writes(rz.{zr, ze, zwrate, zm, zetot})
do
  for z in rz do
    var zr = rinit
    var ze = einit

    var eps = 1e-12
    if z.zx.x > subregion_x0 - eps and
      z.zx.x < subregion_x1 + eps and
      z.zx.y > subregion_y0 - eps and
      z.zx.y < subregion_y1 + eps
    then
      zr = rinitsub
      ze = einitsub
    end

    var zm = zr * z.zvol

    z.zr = zr
    z.ze = ze
    z.zwrate = 0.0
    z.zm = zm
    z.zetot = ze * zm
  end
end

task init_radial_velocity(rp : region(point), vel : double)
where
  reads(rp.px),
  writes(rp.pu)
do
  for p in rp do
    if vel == 0.0 then
      p.pu = {x = 0.0, y = 0.0}
    else
      var pmag = length(p.px)
      p.pu = (vel / pmag)*p.px
    end
  end
end

-- #####################################
-- ## Main simulation loop
-- #################

-- Save off point variable values from previous cycle.
task init_step_points(rp : region(point),
                      enable : bool)
where
  writes(rp.{pmaswt, pf})
do
  if not enable then return end

  -- Initialize fields used in reductions.
  __demand(__vectorize)
  for p in rp do
    p.pmaswt = 0.0
  end
  __demand(__vectorize)
  for p in rp do
    p.pf.x = 0.0
  end
  __demand(__vectorize)
  for p in rp do
    p.pf.y = 0.0
  end
end

--
-- 1. Advance mesh to center of time step.
--
task adv_pos_half(rp : region(point), dt : double,
                  enable : bool)
where
  reads(rp.{px, pu}),
  writes(rp.{px0, pxp, pu0})
do
  if not enable then return end

  var dth = 0.5 * dt

  -- Copy state variables from previous time step and update position.
  __demand(__vectorize)
  for p in rp do
    var px0_x = p.px.x
    var pu0_x = p.pu.x
    p.px0.x = px0_x
    p.pu0.x = pu0_x
    p.pxp.x = px0_x + dth*pu0_x
  end
  __demand(__vectorize)
  for p in rp do
    var px0_y = p.px.y
    var pu0_y = p.pu.y
    p.px0.y = px0_y
    p.pu0.y = pu0_y
    p.pxp.y = px0_y + dth*pu0_y
  end
end

-- Save off zone variable value from previous cycle.
task init_step_zones(rz : region(zone),
                     use_foreign : bool, enable : bool)
where
  reads(rz.zvol),
  writes(rz.zvol0)
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_init_step_zones(zstart, zend, rz)
  else
    -- Copy state variables from previous time step.
    __demand(__vectorize)
    for z in rz do
      z.zvol0 = z.zvol
    end
  end
end

--
-- 1a. Compute new mesh geometry.
--

-- Compute centers of zones and edges.
task calc_centers(rz : region(zone), rpp : region(point), rpg : region(point),
                  rs : region(side(rz, rpp, rpg, rs)),
                  use_foreign : bool, enable : bool)
where
  reads(rz.znump, rpp.pxp, rpg.pxp, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.zxp, rs.exp)
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_centers(sstart, send, rz, rpp, rpg, rs)
  else
    var zxp = vec2 { x = 0.0, y = 0.0 }
    var nside = 1
    for s in rs do
      var z = s.mapsz
      var p1 = s.mapsp1
      var p2 = s.mapsp2
      var e = s

      var p1_pxp = p1.pxp
      e.exp = 0.5*(p1_pxp + p2.pxp)

      zxp += p1_pxp

      if nside == z.znump then
        z.zxp = (1/double(z.znump)) * zxp
        zxp = vec2 { x = 0.0, y = 0.0 }
        nside = 0
      end
      nside += 1
    end
  end
end

-- Compute volumes of zones and sides.
-- Compute edge lengths.
task calc_volumes(rz : region(zone), rpp : region(point), rpg : region(point),
                  rs : region(side(rz, rpp, rpg, rs)),
                  use_foreign : bool, enable : bool)
where
  reads(rz.{zxp, znump}, rpp.pxp, rpg.pxp, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.{zareap, zvolp}, rs.{sareap, elen})
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_volumes(sstart, send, rz, rpp, rpg, rs)
  else
    var zareap = 0.0
    var zvolp = 0.0
    var nside = 1
    for s in rs do
      var z = s.mapsz
      var p1 = s.mapsp1
      var p2 = s.mapsp2

      var p1_pxp = p1.pxp
      var p2_pxp = p2.pxp
      var sa = 0.5 * cross(p2_pxp - p1_pxp, z.zxp - p1_pxp)
      var sv = sa * (p1_pxp.x + p2_pxp.x + z.zxp.x)
      s.sareap = sa
      -- s.svolp = sv
      s.elen = length(p2_pxp - p1_pxp)

      zareap += sa
      zvolp += sv

      if nside == z.znump then
        z.zareap = zareap
        z.zvolp = (1.0 / 3.0) * zvolp
        zareap = 0.0
        zvolp = 0.0
        nside = 0
      end
      nside += 1

      regentlib.assert(sv > 0.0, "sv negative")
    end
  end
end

-- Compute zone characteristic lengths.
task calc_char_len(rz : region(zone), rpp : region(point), rpg : region(point),
                   rs : region(side(rz, rpp, rpg, rs)),
                   use_foreign : bool, enable : bool)
where
  reads(rz.znump, rs.{mapsz, sareap, elen}),
  writes(rz.zdl)
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_char_len(sstart, send, rz, rs)
  else
    var zdl = 1e99
    var nside = 1
    for s in rs do
      var z = s.mapsz
      var e = s

      var area = s.sareap
      var base = e.elen
      var fac = 0.0
      if z.znump == 3 then
        fac = 3.0
      else
        fac = 4.0
      end
      var sdl = fac * area / base
      zdl = min(zdl, sdl)

      if nside == z.znump then
        z.zdl = zdl
        zdl = 1e99
        nside = 0
      end
      nside += 1
    end
  end
end

--
-- 2. Compute point masses.
--

-- Compute zone densities.
task calc_rho_half(rz : region(zone),
                   use_foreign : bool, enable : bool)
where
  reads(rz.{zvolp, zm}),
  writes(rz.zrp)
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_rho_half(zstart, zend, rz)
  else
    __demand(__vectorize)
    for z in rz do
      z.zrp = z.zm / z.zvolp
    end
  end
end

-- Reduce masses into points.
task sum_point_mass(rz : region(zone), rpp : region(point), rpg : region(point),
                    rs : region(side(rz, rpp, rpg, rs)),
                    use_foreign : bool, enable : bool)
where
  reads(rz.{zareap, zrp}, rs.{mapsz, mapsp1, mapss3, smf}),
  reads writes(rpp.pmaswt),
  reduces+(rpg.pmaswt)
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_sum_point_mass(sstart, send, rz, rpp, rpg, rs)
  else
    for s in rs do
      var z = s.mapsz
      var p1 = s.mapsp1
      var s3 = s.mapss3

      var m = z.zrp * z.zareap * 0.5 * (s.smf + s3.smf)
      p1.pmaswt += m
    end
  end
end

--
-- 3. Compute material state (half-advanced).
--

task calc_state_at_half(rz : region(zone),
                        gamma : double, ssmin : double, dt : double,
                        use_foreign : bool, enable : bool)
where
  reads(rz.{zvol0, zvolp, zm, zr, ze, zwrate}),
  writes(rz.{zp, zss})
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_state_at_half(gamma, ssmin, dt, zstart, zend, rz)
  else
    var gm1 = gamma - 1.0
    var ss2 = max(ssmin * ssmin, 1e-99)
    var dth = 0.5 * dt

    for z in rz do
      var rx = z.zr
      var ex = max(z.ze, 0.0)
      var px = gm1 * rx * ex
      var prex = gm1 * ex
      var perx = gm1 * rx
      var csqd = max(ss2, prex + perx * px / (rx * rx))
      var z0per = perx
      var zss = sqrt(csqd)
      z.zss = zss

      var zminv = 1.0 / z.zm
      var dv = (z.zvolp - z.zvol0) * zminv
      var bulk = z.zr * zss * zss
      var denom = 1.0 + 0.5 * z0per * dv
      var src = z.zwrate * dth * zminv
      z.zp = px + (z0per * src - z.zr * bulk * dv) / denom
    end
  end
end

--
-- 4. Compute forces.
--

-- Compute PolyGas and TTS forces.
task calc_force_pgas_tts(rz : region(zone), rpp : region(point),
                         rpg : region(point),
                         rs : region(side(rz, rpp, rpg, rs)),
                         alfa : double, ssmin : double,
                         use_foreign : bool, enable : bool)
where
  reads(rz.{zxp, zareap, zrp, zss, zp}, rs.{mapsz, sareap, smf, exp}),
  writes(rs.{sfp, sft})
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_force_pgas(sstart, send, rz, rs)
  else
    for s in rs do
      var z = s.mapsz

      -- Compute surface vectors of sides.
      var ssurfp = rotateCCW(s.exp - z.zxp)

      -- Compute PolyGas forces.
      var sfx = (-z.zp)*ssurfp
      s.sfp = sfx

      -- Compute TTS forces.
      var svfacinv = z.zareap / s.sareap
      var srho = z.zrp * s.smf * svfacinv
      var sstmp = max(z.zss, ssmin)
      sstmp = alfa * sstmp * sstmp
      var sdp = sstmp * (srho - z.zrp)
      var sqq = (-sdp)*ssurfp
      s.sft = sfx + sqq
    end
  end
end

task qcs_zone_center_velocity(rz : region(zone), rpp : region(point), rpg : region(point),
                              rs : region(side(rz, rpp, rpg, rs)),
                              use_foreign : bool, enable : bool)
where
  reads(rz.znump, rpp.pu, rpg.pu, rs.{mapsz, mapsp1}),
  writes(rz.zuc)
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_qcs_zone_center_velocity(sstart, send, rz, rpp, rpg, rs)
  else
    var zuc = vec2 { x = 0.0, y = 0.0 }
    var nside = 1
    for s in rs do
      var z = s.mapsz
      var p1 = s.mapsp1

      zuc += (1.0 / double(z.znump))*p1.pu

      if nside == z.znump then
        z.zuc = zuc
        zuc = vec2 { x = 0.0, y = 0.0 }
        nside = 0
      end
      nside += 1
    end
  end
end

terra foreign_qcs_corner_divergence(
  rz_physical : c.legion_physical_region_t[4],
  rz_fields : c.legion_field_id_t[4],
  rpp_physical : c.legion_physical_region_t[4],
  rpp_fields : c.legion_field_id_t[4],
  rpg_physical : c.legion_physical_region_t[4],
  rpg_fields : c.legion_field_id_t[4],
  rs_physical : c.legion_physical_region_t[14],
  rs_fields : c.legion_field_id_t[14])
end

task qcs_corner_divergence(rz : region(zone), rpp : region(point), rpg : region(point),
                           rs : region(side(rz, rpp, rpg, rs)),
                           use_foreign : bool, enable : bool)
where
  reads(rz.{zxp, zuc}, rpp.{pxp, pu}, rpg.{pxp, pu},
        rs.{mapsz, mapsp1, mapsp2, mapss3, exp, elen}),
  writes(rs.{carea, ccos, cdiv, cevol, cdu})
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    foreign_qcs_corner_divergence(
      __physical(rz), __fields(rz),
      __physical(rpp), __fields(rpp),
      __physical(rpg), __fields(rpg),
      __physical(rs), __fields(rs))
  else
    for s2 in rs do
      var c = s2
      var s = s2.mapss3
      var z = s.mapsz
      var p = s.mapsp2
      var p1 = s.mapsp1
      var p2 = s2.mapsp2
      var e1 = s
      var e2 = s2

      -- velocities and positions
      -- point p
      var up0 = p.pu
      var xp0 = p.pxp
      -- edge e2
      var up1 = 0.5*(p.pu + p2.pu)
      var xp1 = e2.exp
      -- zone center z
      var up2 = z.zuc
      var xp2 = z.zxp
      -- edge e1
      var up3 = 0.5*(p1.pu + p.pu)
      var xp3 = e1.exp

      -- compute 2d cartesian volume of corner
      var cvolume = 0.5 * cross(xp2 - xp0, xp3 - xp1)
      c.carea = cvolume

      -- compute cosine angle
      var v1 = xp3 - xp0
      var v2 = xp1 - xp0
      var de1 = e1.elen
      var de2 = e2.elen
      var minelen = min(de1, de2)
      if minelen < 1e-12 then
        c.ccos = 0.0
      else
        c.ccos = 4.0 * dot(v1, v2) / (de1 * de2)
      end

      -- compute divergence of corner
      var cdiv = (cross(up2 - up0, xp3 - xp1) -
                  cross(up3 - up1, xp2 - xp0)) / (2.0 * cvolume)
      c.cdiv = cdiv

      -- compute evolution factor
      var dxx1 = 0.5*(((xp1 + xp2) - xp0) - xp3)
      var dxx2 = 0.5*(((xp2 + xp3) - xp0) - xp1)
      var dx1 = length(dxx1)
      var dx2 = length(dxx2)

      -- average corner-centered velocity
      var duav = 0.25*(((up0 + up1) + up2) + up3)

      var test1 = abs(dot(dxx1, duav) * dx2)
      var test2 = abs(dot(dxx2, duav) * dx1)
      var num = 0.0
      var den = 0.0
      if test1 > test2 then
        num = dx1
        den = dx2
      else
        num = dx2
        den = dx1
      end
      var r = num / den
      var evol = min(sqrt(4.0 * cvolume * r), 2.0 * minelen)

      -- compute delta velocity
      var dv1 = length(((up1 + up2) - up0) - up3)
      var dv2 = length(((up2 + up3) - up0) - up1)
      var du = max(dv1, dv2)

      if cdiv < 0.0 then
        c.cevol = evol
        c.cdu = du
      else
        c.cevol = 0.0
        c.cdu = 0.0
      end
    end
  end
end

terra foreign_qcs_qcn_force(
  rz_physical : c.legion_physical_region_t[2],
  rz_fields : c.legion_field_id_t[2],
  rpp_physical : c.legion_physical_region_t[2],
  rpp_fields : c.legion_field_id_t[2],
  rpg_physical : c.legion_physical_region_t[2],
  rpg_fields : c.legion_field_id_t[2],
  rs_physical : c.legion_physical_region_t[14],
  rs_fields : c.legion_field_id_t[14])
end

task qcs_qcn_force(rz : region(zone), rpp : region(point), rpg : region(point),
                   rs : region(side(rz, rpp, rpg, rs)),
                   gamma : double, q1 : double, q2 : double,
                   use_foreign : bool, enable : bool)
where
  reads(rz.{zrp, zss}, rpp.pu, rpg.pu,
        rs.{mapsz, mapsp1, mapsp2, mapss3, elen, cdiv, cdu, cevol}),
  writes(rs.{cqe1, cqe2})
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    foreign_qcs_qcn_force(
      __physical(rz), __fields(rz),
      __physical(rpp), __fields(rpp),
      __physical(rpg), __fields(rpg),
      __physical(rs), __fields(rs))
  else
    var gammap1 = gamma + 1.0

    for s4 in rs do
      var c = s4
      var z = c.mapsz

      var ztmp2 = q2 * 0.25 * gammap1 * c.cdu
      var ztmp1 = q1 * z.zss
      var zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1)
      var rmu = zkur * z.zrp * c.cevol
      if c.cdiv > 0.0 then
        rmu = 0.0
      end

      var s = c.mapss3
      var p = s.mapsp2
      var p1 = s.mapsp1
      var e1 = s
      var p2 = s4.mapsp2
      var e2 = s4

      c.cqe1 = rmu / e1.elen*(p.pu - p1.pu)
      c.cqe2 = rmu / e2.elen*(p2.pu - p.pu)
    end
  end
end

terra foreign_qcs_force(
  rs_physical : c.legion_physical_region_t[10],
  rs_fields : c.legion_field_id_t[10])
end

task qcs_force(rz : region(zone), rpp : region(point), rpg : region(point),
               rs : region(side(rz, rpp, rpg, rs)),
               use_foreign : bool, enable : bool)
where
  reads(rs.{mapss4, elen, carea, ccos, cqe1, cqe2}),
  writes(rs.sfq)
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    foreign_qcs_force(
      __physical(rs), __fields(rs))
  else
    for s in rs do
      var c1 = s
      var c2 = s.mapss4
      var e = s
      var el = e.elen

      var c1sin2 = 1.0 - c1.ccos * c1.ccos
      var c1w = 0.0
      var c1cos = 0.0
      if c1sin2 >= 1e-4 then
        c1w = c1.carea / c1sin2
        c1cos = c1.ccos
      end

      var c2sin2 = 1.0 - c2.ccos * c2.ccos
      var c2w = 0.0
      var c2cos = 0.0
      if c2sin2 >= 1e-4 then
        c2w = c2.carea / c2sin2
        c2cos = c2.ccos
      end

      s.sfq = (1.0 / el)*(c1w*(c1.cqe2 + c1cos*c1.cqe1) +
                            c2w*(c2.cqe1 + c2cos*c2.cqe2))
    end
  end
end

task qcs_vel_diff(rz : region(zone), rpp : region(point), rpg : region(point),
                  rs : region(side(rz, rpp, rpg, rs)),
                  q1 : double, q2 : double,
                  use_foreign : bool, enable : bool)
where
  reads(rz.{zss, z0tmp}, rpp.{pxp, pu}, rpg.{pxp, pu},
        rs.{mapsp1, mapsp2, mapsz, elen}),
  writes(rz.{zdu, z0tmp})
do
  if not enable then return end

  for z in rz do
    z.z0tmp = 0.0
  end

  for s in rs do
    var p1 = s.mapsp1
    var p2 = s.mapsp2
    var z = s.mapsz
    var e = s

    var dx = p2.pxp - p1.pxp
    var du = p2.pu - p1.pu
    var lenx = e.elen
    var dux = dot(du, dx)
    if lenx > 0.0 then
      dux = abs(dux) / lenx
    else
      dux = 0.0
    end
    z.z0tmp = max(z.z0tmp, dux)
  end

  for z in rz do
    z.zdu = q1 * z.zss + 2.0 * q2 * z.z0tmp
  end
end

-- Reduce forces into points.
task sum_point_force(rz : region(zone), rpp : region(point), rpg : region(point),
                     rs : region(side(rz, rpp, rpg, rs)),
                     use_foreign : bool, enable : bool)
where
  reads(rz.znump, rs.{mapsz, mapsp1, mapss3, sfq, sft}),
  reads writes(rpp.pf),
  reduces+(rpg.pf.{x, y})
do
  if not enable then return end

  if use_foreign then
    -- foreign_sum_point_force(
    --   __physical(rz), __fields(rz),
    --   __physical(rpp), __fields(rpp),
    --   __physical(rpg), __fields(rpg),
    --   __physical(rs), __fields(rs))
  else
    for s in rs do
      var p1 = s.mapsp1
      var s3 = s.mapss3

      var f = (s.sfq + s.sft) - (s3.sfq + s3.sft)
      p1.pf.x += f.x
      p1.pf.y += f.y
    end
  end
end

--
-- 4a. Apply boundary conditions.
--

task apply_boundary_conditions(rp : region(point),
                               enable : bool)
where
  reads(rp.{has_bcx, has_bcy}),
  reads writes(rp.{pu0, pf})
do
  if not enable then return end

  var vfixx = {x = 1.0, y = 0.0}
  var vfixy = {x = 0.0, y = 1.0}
  for p in rp do
    if p.has_bcx then
      p.pu0 = project(p.pu0, vfixx)
      p.pf = project(p.pf, vfixx)
    end
    if p.has_bcy then
      p.pu0 = project(p.pu0, vfixy)
      p.pf = project(p.pf, vfixy)
    end
  end
end

--
-- 5. Compute accelerations.
--

-- Fused into adv_pos_full.

--
-- 6. Advance mesh to end of time step.
--

task adv_pos_full(rp : region(point), dt : double,
                  enable : bool)
where
  reads(rp.{px0, pu0, pf, pmaswt}),
  writes(rp.{px, pu})
do
  if not enable then return end

  var fuzz = 1e-99
  var dth = 0.5 * dt
  __demand(__vectorize)
  for p in rp do
    var fac = 1.0 / max(p.pmaswt, fuzz)
    var pap_x = fac*p.pf.x
    var pap_y = fac*p.pf.y

    var pu_x = p.pu0.x + dt*(pap_x)
    p.pu.x = pu_x
    p.px.x = p.px0.x + dth*(pu_x + p.pu0.x)

    var pu_y = p.pu0.y + dt*(pap_y)
    p.pu.y = pu_y
    p.px.y = p.px0.y + dth*(pu_y + p.pu0.y)
  end
end

--
-- 6a. Compute new mesh geometry.
--

-- FIXME: This is a duplicate of calc_centers but with different
-- code. Struct slicing ought to make it possible to use the same code
-- in both cases.
task calc_centers_full(rz : region(zone), rpp : region(point), rpg : region(point),
                       rs : region(side(rz, rpp, rpg, rs)),
                       use_foreign : bool, enable : bool)
where
  reads(rz.znump, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.zx, rs.ex)
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_centers_full(sstart, send, rz, rpp, rpg, rs)
  else
    var zx = vec2 { x = 0.0, y = 0.0 }
    var nside = 1
    for s in rs do
      var z = s.mapsz
      var p1 = s.mapsp1
      var p2 = s.mapsp2
      var e = s

      var p1_px = p1.px
      e.ex = 0.5*(p1_px + p2.px)

      zx += p1_px

      if nside == z.znump then
        z.zx = (1/double(z.znump)) * zx
        zx = vec2 { x = 0.0, y = 0.0 }
        nside = 0
      end
      nside += 1
    end
  end
end

-- FIXME: This is a duplicate of calc_volumes but with different
-- code. Struct slicing ought to make it possible to use the same code
-- in both cases.
task calc_volumes_full(rz : region(zone), rpp : region(point), rpg : region(point),
                       rs : region(side(rz, rpp, rpg, rs)),
                       use_foreign : bool, enable : bool)
where
  reads(rz.{zx, znump}, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.{zarea, zvol}, rs.{sarea})
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_volumes_full(sstart, send, rz, rpp, rpg, rs)
  else
    var zarea = 0.0
    var zvol = 0.0
    var nside = 1
    for s in rs do
      var z = s.mapsz
      var p1 = s.mapsp1
      var p2 = s.mapsp2

      var p1_px = p1.px
      var p2_px = p2.px
      var sa = 0.5 * cross(p2_px - p1_px, z.zx - p1_px)
      var sv = sa * (p1_px.x + p2_px.x + z.zx.x)
      s.sarea = sa
      -- s.svol = sv

      zarea += sa
      zvol += sv

      if nside == z.znump then
        z.zarea = zarea
        z.zvol = (1.0 / 3.0) * zvol
        zarea = 0.0
        zvol = 0.0
        nside = 0
      end
      nside += 1

      regentlib.assert(sv > 0.0, "sv negative")
    end
  end
end

--
-- 7. Compute work
--

task calc_work(rz : region(zone), rpp : region(point), rpg : region(point),
               rs : region(side(rz, rpp, rpg, rs)),
               dt : double,
               use_foreign : bool, enable : bool)
where
  reads(rz.{zetot, znump}, rpp.{pxp, pu0, pu}, rpg.{pxp, pu0, pu},
        rs.{mapsz, mapsp1, mapsp2, sfp, sfq}),
  writes(rz.{zw, zetot})
do
  if not enable then return end

  if false --[[ use_foreign ]] then
    -- foreign_calc_work(dt, sstart, send, rz, rpp, rpg, rs)
  else
    var zdwork = 0.0
    var nside = 1
    for s in rs do
      var z = s.mapsz
      var p1 = s.mapsp1
      var p2 = s.mapsp2

      var sftot = s.sfp + s.sfq
      var sd1 = dot(sftot, p1.pu0 + p1.pu)
      var sd2 = dot(-1.0*sftot, p2.pu0 + p2.pu)
      var dwork = -0.5 * dt * (sd1 * p1.pxp.x + sd2 * p2.pxp.x)

      zdwork += dwork

      if nside == z.znump then
        z.zetot += zdwork
        z.zw = zdwork
        zdwork = 0.0
        nside = 0
      end
      nside += 1
    end
  end
end

--
-- 7a. Compute work rate.
-- 8. Update state variables.
--

task calc_work_rate_energy_rho_full(rz : region(zone), dt : double,
                                    enable : bool)
where
  reads(rz.{zvol0, zvol, zm, zw, zp, zetot}),
  writes(rz.{zwrate, ze, zr})
do
  if not enable then return end

  var dtiny = 1.0 / dt
  var fuzz = 1e-99

  __demand(__vectorize)
  for z in rz do
    var dvol = z.zvol - z.zvol0
    z.zwrate = (z.zw + z.zp * dvol) * dtiny

    z.ze = z.zetot / (z.zm + fuzz)

    z.zr = z.zm / z.zvol
  end
end

--
-- 9. Compute timstep for next cycle.
--

--[[
task calc_dt_courant(rz : region(zone), dtmax : double, cfl : double) : double
where
  reads(rz.{zdl, zss, zdu})
do
  var fuzz = 1e-99
  var dtnew = dtmax
  for z in rz do
    var cdu = max(z.zdu, max(z.zss, fuzz))
    var zdthyd = z.zdl * cfl / cdu

    dtnew min= zdthyd
  end

  return dtnew
end

task calc_dt_volume(rz : region(zone), dtlast : double, cflv : double) : double
where
  reads(rz.{zvol0, zvol})
do
  var dvovmax = 1e-99
  for z in rz do
    var zdvov = abs((z.zvol - z.zvol0) / z.zvol0)
    dvovmax max= zdvov
  end
  return dtlast * cflv / dvovmax
end
]]

task calc_dt_hydro(rz : region(zone), dtlast : double, dtmax : double,
                   cfl : double, cflv : double, enable : bool) : double
where
  reads(rz.{zdl, zvol0, zvol, zss, zdu})
do
  var dthydro = dtmax

  if not enable then return dthydro end

  -- dthydro min= min(calc_dt_courant(rz, dtmax, cfl),
  --                  calc_dt_volume(rz, dtlast, cflv))

  -- Hack: manually inline calc_dt_courant
  do
    var fuzz = 1e-99
    var dtnew = dtmax
    for z in rz do
      var cdu = max(z.zdu, max(z.zss, fuzz))
      var zdthyd = z.zdl * cfl / cdu

      dtnew min= zdthyd
    end

    dthydro min= dtnew
  end

  -- Hack: manually inline calc_dt_volume
  do
    var dvovmax = 1e-99
    for z in rz do
      var zdvov = abs((z.zvol - z.zvol0) / z.zvol0)
      dvovmax max= zdvov
    end
    dthydro min= dtlast * cflv / dvovmax
  end

  return dthydro
end

task calc_global_dt(dt : double, dtfac : double, dtinit : double,
                    dtmax : double, dthydro : double,
                    time : double, tstop : double, cycle : int64) : double
  var dtlast = dt

  dt = dtmax

  if cycle == 0 then
    dt = min(dt, dtinit)
  else
    var dtrecover = dtfac * dtlast
    dt = min(dt, dtrecover)
  end

  dt = min(dt, tstop - time)
  dt = min(dt, dthydro)

  return dt
end

function _t2() end -- seems like I need this to break up the statements
(terra() c.printf("Compiling simulation (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6) end)()

task continue_simulation(cycle : int64, cstop : int64,
                         time : double, tstop : double)
  return (cycle < cstop and time < tstop)
end

task simulate(rz_all : region(zone), rz_all_p : partition(disjoint, rz_all),
              rp_all : region(point),
              rp_all_private : region(point),
              rp_all_private_p : partition(disjoint, rp_all_private),
              rp_all_ghost : region(point),
              rp_all_ghost_p : partition(aliased, rp_all_ghost),
              rp_all_shared_p : partition(disjoint, rp_all_ghost),
              rs_all : region(side(wild, wild, wild, wild)),
              rs_all_p : partition(disjoint, rs_all),
              conf : config)
where
  reads writes(rz_all, rp_all_private, rp_all_ghost, rs_all),
  rp_all_private * rp_all_ghost
do
  var alfa = conf.alfa
  var cfl = conf.cfl
  var cflv = conf.cflv
  var cstop = conf.cstop
  var dtfac = conf.dtfac
  var dtinit = conf.dtinit
  var dtmax = conf.dtmax
  var gamma = conf.gamma
  var q1 = conf.q1
  var q2 = conf.q2
  var qgamma = conf.qgamma
  var ssmin = conf.ssmin
  var tstop = conf.tstop
  var uinitradial = conf.uinitradial
  var vfix = {x = 0.0, y = 0.0}

  var use_foreign = conf.use_foreign
  var enable = conf.enable

  var interval = 10
  var start_time = c.legion_get_current_time_in_micros()/1.e6
  var last_time = start_time

  var time = 0.0
  var cycle : int64 = 0
  var dt = dtmax
  var dthydro = dtmax
  while continue_simulation(cycle, cstop, time, tstop) do
    c.legion_runtime_begin_trace(__runtime(), __context(), 0)

    __demand(__parallel)
    for i = 0, conf.npieces do
      init_step_points(rp_all_private_p[i], enable)
    end
    __demand(__parallel)
    for i = 0, conf.npieces do
      init_step_points(rp_all_shared_p[i], enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      init_step_zones(rz_all_p[i], use_foreign, enable)
    end

    dt = calc_global_dt(dt, dtfac, dtinit, dtmax, dthydro, time, tstop, cycle)

    if cycle > 0 and cycle % interval == 0 then
      var current_time = c.legion_get_current_time_in_micros()/1.e6
      c.printf("cycle %4ld    sim time %.3e    dt %.3e    time %.3e (per iteration) %.3e (total)\n",
               cycle, time, dt, (current_time - last_time)/interval, current_time - start_time)
      last_time = current_time
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      adv_pos_half(rp_all_private_p[i], dt, enable)
    end
    __demand(__parallel)
    for i = 0, conf.npieces do
      adv_pos_half(rp_all_shared_p[i], dt, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_centers(rz_all_p[i],
                   rp_all_private_p[i],
                   rp_all_ghost_p[i],
                   rs_all_p[i],
                   use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_volumes(rz_all_p[i],
                   rp_all_private_p[i],
                   rp_all_ghost_p[i],
                   rs_all_p[i],
                   use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_char_len(rz_all_p[i],
                    rp_all_private_p[i],
                    rp_all_ghost_p[i],
                    rs_all_p[i],
                    use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_rho_half(rz_all_p[i], use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      sum_point_mass(rz_all_p[i],
                     rp_all_private_p[i],
                     rp_all_ghost_p[i],
                     rs_all_p[i],
                     use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_state_at_half(rz_all_p[i], gamma, ssmin, dt, use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_force_pgas_tts(rz_all_p[i],
                          rp_all_private_p[i],
                          rp_all_ghost_p[i],
                          rs_all_p[i],
                          alfa, ssmin,
                          use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_zone_center_velocity(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_corner_divergence(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_qcn_force(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        gamma, q1, q2,
        use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_force(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      qcs_vel_diff(
        rz_all_p[i],
        rp_all_private_p[i],
        rp_all_ghost_p[i],
        rs_all_p[i],
        q1, q2,
        use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      sum_point_force(rz_all_p[i],
                      rp_all_private_p[i],
                      rp_all_ghost_p[i],
                      rs_all_p[i],
                      use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      apply_boundary_conditions(rp_all_private_p[i], enable)
    end
    __demand(__parallel)
    for i = 0, conf.npieces do
      apply_boundary_conditions(rp_all_shared_p[i], enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      adv_pos_full(rp_all_private_p[i], dt, enable)
    end
    __demand(__parallel)
    for i = 0, conf.npieces do
      adv_pos_full(rp_all_shared_p[i], dt, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_centers_full(rz_all_p[i],
                        rp_all_private_p[i],
                        rp_all_ghost_p[i],
                        rs_all_p[i],
                        use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_volumes_full(rz_all_p[i],
                        rp_all_private_p[i],
                        rp_all_ghost_p[i],
                        rs_all_p[i],
                        use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_work(rz_all_p[i],
                rp_all_private_p[i],
                rp_all_ghost_p[i],
                rs_all_p[i],
                dt, use_foreign, enable)
    end

    __demand(__parallel)
    for i = 0, conf.npieces do
      calc_work_rate_energy_rho_full(rz_all_p[i], dt, enable)
    end

    dthydro = dtmax
    __demand(__parallel)
    for i = 0, conf.npieces do
      dthydro min= calc_dt_hydro(rz_all_p[i], dt, dtmax, cfl, cflv, enable)
    end

    cycle += 1
    time += dt

    c.legion_runtime_end_trace(__runtime(), __context(), 0)
  end
end

task initialize(rz_all : region(zone), rz_all_p : partition(disjoint, rz_all),
                rp_all : region(point),
                rp_all_private : region(point),
                rp_all_private_p : partition(disjoint, rp_all_private),
                rp_all_ghost : region(point),
                rp_all_ghost_p : partition(aliased, rp_all_ghost),
                rp_all_shared_p : partition(disjoint, rp_all_ghost),
                rs_all : region(side(wild, wild, wild, wild)),
                rs_all_p : partition(disjoint, rs_all),
                conf : config)
where
  reads writes(rz_all, rp_all_private, rp_all_ghost, rs_all),
  rp_all_private * rp_all_ghost
do
  var einit = conf.einit
  var einitsub = conf.einitsub
  var rinit = conf.rinit
  var rinitsub = conf.rinitsub
  var subregion = conf.subregion
  var uinitradial = conf.uinitradial

  var use_foreign = conf.use_foreign
  var enable = true

  for i = 0, conf.npieces do
    init_mesh_zones(rz_all_p[i])
  end

  for i = 0, conf.npieces do
    calc_centers_full(rz_all_p[i],
                      rp_all_private_p[i],
                      rp_all_ghost_p[i],
                      rs_all_p[i],
                      use_foreign, enable)
  end

  for i = 0, conf.npieces do
    calc_volumes_full(rz_all_p[i],
                      rp_all_private_p[i],
                      rp_all_ghost_p[i],
                      rs_all_p[i],
                      use_foreign, enable)
  end

  for i = 0, conf.npieces do
    init_side_fracs(rz_all_p[i],
                    rp_all_private_p[i],
                    rp_all_ghost_p[i],
                    rs_all_p[i])
  end

  for i = 0, conf.npieces do
    init_hydro(rz_all_p[i],
               rinit, einit, rinitsub, einitsub,
               subregion[0], subregion[1], subregion[2], subregion[3])
  end

  for i = 0, conf.npieces do
    init_radial_velocity(rp_all_private_p[i], uinitradial)
    init_radial_velocity(rp_all_shared_p[i], uinitradial)
  end

  if conf.warmup then
    -- Do one iteration to warm up the runtime.
    var conf_warmup = conf
    conf_warmup.cstop = 1
    conf_warmup.enable = false
    simulate(rz_all, rz_all_p,
             rp_all,
             rp_all_private, rp_all_private_p,
             rp_all_ghost, rp_all_ghost_p, rp_all_shared_p,
             rs_all, rs_all_p,
             conf_warmup)
  end
end

task dummy(rz : region(zone)) : int
where reads(rz) do
  return 1
end

terra wait_for(x : int)
  return x
end

function _t20() end -- seems like I need this to break up the statements
(terra() c.printf("Compiling main (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6) end)()

--
-- Command Line Processing
--

terra get_positional_arg()
  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if args.argv[i][0] == ('-')[0] then
      i = i + 1
    else
      return args.argv[i]
    end
    i = i + 1
  end
  return nil
end

terra get_optional_arg(key : rawstring)
  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if cstring.strcmp(args.argv[i], key) == 0 then
      if i + 1 < args.argc then
        return args.argv[i + 1]
      else
        return nil
      end
    elseif args.argv[i][0] == ('-')[0] then
      i = i + 1
    end
    i = i + 1
  end
  return nil
end

--
-- Configuration
--

terra init_mesh(conf : &config)
  conf.nzx = conf.meshparams[0]
  if conf.meshparams_n >= 2 then
    conf.nzy = conf.meshparams[1]
  else
    conf.nzy = conf.nzx
  end
  if conf.meshtype ~= MESH_PIE then
    if conf.meshparams_n >= 3 then
      conf.lenx = conf.meshparams[2]
    else
      conf.lenx = 1.0
    end
  else
    -- convention:  x = theta, y = r
    if conf.meshparams_n >= 3 then
      conf.lenx = conf.meshparams[2] * cmath.M_PI / 180.0
    else
      conf.lenx = 90.0 * cmath.M_PI / 180.0
    end
  end
  if conf.meshparams_n >= 4 then
    conf.leny = conf.meshparams[3]
  else
    conf.leny = 1.0
  end

  if conf.nzx <= 0 or conf.nzy <= 0 or conf.lenx <= 0. or conf.leny <= 0. then
    c.printf("Error: meshparams values must be positive\n")
    c.abort()
  end
  if conf.meshtype == MESH_PIE and conf.lenx >= 2. * cmath.M_PI then
    c.printf("Error: meshparams theta must be < 360\n")
    c.abort()
  end

  -- Calculate approximate nz, np, ns for region size upper bound.
  conf.nz = conf.nzx * conf.nzy
  conf.np = (conf.nzx + 1) * (conf.nzy + 1)
  if conf.meshtype ~= MESH_HEX then
    conf.maxznump = 4
  else
    conf.maxznump = 6
  end
  conf.ns = conf.nz * conf.maxznump
end

do
local max_items = 1024
local max_item_len = 1024
local fixed_string = int8[max_item_len]

function get_type_specifier(t, as_input)
  if t == fixed_string then
    if as_input then
      return "%" .. max_item_len .. "s", 1
    else
      return "%s"
    end
  elseif t == int64 then
    return "%lld", 1
  elseif t == bool then
    return "%d", 1
  elseif t == double then
    if as_input then
      return "%lf", 1
    else
      return "%.2e", 1
    end
  elseif t:isarray() then
    local elt_type_specifier = get_type_specifier(t.type, as_input)
    local type_specifier = ""
    for i = 1, t.N do
      if i > 1 then
        type_specifier = type_specifier .. " "
      end
      type_specifier = type_specifier .. elt_type_specifier
    end
    return type_specifier, t.N
  else
    assert(false)
  end
end

function explode_array(t, value)
  if t == fixed_string then
    return terralib.newlist({`@value})
  elseif t:isarray() then
    local values = terralib.newlist()
    for i = 0, t.N - 1 do
      values:insert(`&((@value)[i]))
    end
    return values
  else
    return terralib.newlist({value})
  end
end

local extract = terralib.memoize(function(t)
  local str_specifier = get_type_specifier(fixed_string, true)
  local type_specifier, size = get_type_specifier(t, true)

  return terra(items : &(fixed_string), nitems : int64, key : rawstring, result : &t)
    var item_key : fixed_string
    for i = 0, nitems do
      var matched = c.sscanf(&(items[i][0]), [str_specifier], item_key)
      if matched >= 1 and cstring.strncmp(key, item_key, max_item_len) == 0 then
        var matched = c.sscanf(
          &(items[i][0]), [str_specifier .. " " .. type_specifier],
          item_key, [explode_array(t, result)])
        if matched >= 1 then
          return matched - 1
        end
      end
    end
  end
end)

terra read_config()
  var input_filename = get_positional_arg()
  if input_filename == nil then
    c.printf("Usage: ./pennant <filename>\n")
    c.abort()
  end

  c.printf("Reading \"%s\"...\n", input_filename)
  var input_file = c.fopen(input_filename, "r")
  if input_file == nil then
    c.printf("Error: Failed to open \"%s\"\n", input_filename)
    c.abort()
  end

  var items : fixed_string[max_items]

  var nitems = 0
  for i = 0, max_items do
    if c.fgets(items[i], max_item_len, input_file) == nil then
      nitems = i + 1
      break
    end
  end

  if c.fclose(input_file) ~= 0 then
    c.printf("Error: Failed to close \"%s\"\n", input_filename)
    c.abort()
  end

  var conf : config

  -- Set defaults.
  [config_fields_all:map(function(field)
       return quote conf.[field.field] = [field.default_value] end
     end)]

  -- Read parameters from command line.
  var npieces = get_optional_arg("-npieces")
  if npieces ~= nil then
    conf.npieces = c.atoll(npieces)
  end
  if conf.npieces <= 0 then
    c.printf("Error: npieces (%lld) must be >= 0\n", conf.npieces)
    c.abort()
  end

  var use_foreign = get_optional_arg("-foreign")
  if use_foreign ~= nil then
    conf.use_foreign = [bool](c.atoll(use_foreign))
  end

  var warmup = get_optional_arg("-warmup")
  if warmup ~= nil then
    conf.warmup = [bool](c.atoll(warmup))
  end

  var nocheck = get_optional_arg("-nocheck")
  if nocheck ~= nil then
    conf.nocheck = [bool](c.atoll(nocheck))
  end

  var compact = get_optional_arg("-compact")
  if compact ~= nil then
    conf.compact = [bool](c.atoll(compact))
  end

  var stripsize = get_optional_arg("-stripsize")
  if stripsize ~= nil then
    conf.stripsize = c.atoll(stripsize)
  end

  var spansize = get_optional_arg("-spansize")
  if spansize ~= nil then
    conf.spansize = c.atoll(spansize)
  end

  -- Read parameters from input file.
  [config_fields_input:map(function(field)
       if field.is_linked_field then
         return quote end
       else
         if field.linked_field then
           return quote
             conf.[field.linked_field] = [extract(field.type)](items, nitems, field.field, &(conf.[field.field]))
           end
         else
           return quote
             [extract(field.type)](items, nitems, field.field, &(conf.[field.field]))
           end
         end
       end
     end)]

  -- Allow command-line overrides of any file input setting
  [config_fields_input:map(function(field)
       if field.is_linked_field then
         return quote end
       else
         if field.linked_field then
           return quote end
         else
           return quote
	     var argval = get_optional_arg([ "-" .. field.field])
	     if argval ~= nil then
	       c.sscanf(&(argval[0]), [get_type_specifier(field.type, true)],
                        [explode_array(field.type, `(&(conf.[field.field])))])
	     end
           end
         end
       end
     end)]

  -- Configure and run mesh generator.
  var meshtype : fixed_string
  if [extract(fixed_string)](items, nitems, "meshtype", &meshtype) < 1 then
    c.printf("Error: Missing meshtype\n")
    c.abort()
  end
  if cstring.strncmp(meshtype, "pie", max_item_len) == 0 then
    conf.meshtype = MESH_PIE
  elseif cstring.strncmp(meshtype, "rect", max_item_len) == 0 then
    conf.meshtype = MESH_RECT
  elseif cstring.strncmp(meshtype, "hex", max_item_len) == 0 then
    conf.meshtype = MESH_HEX
  else
    c.printf("Error: Invalid meshtype \"%s\"\n", meshtype)
    c.abort()
  end

  c.printf("Config meshtype = \"%s\"\n", meshtype)

  init_mesh(&conf)

  [config_fields_all:map(function(field)
       return quote c.printf(
         ["Config " .. field.field .. " = " .. get_type_specifier(field.type, false) .. "\n"],
         [explode_array(field.type, `&(conf.[field.field])):map(
            function(value)
              return `@value
            end)])
       end
     end)]

  -- report mesh size in bytes
  do
    var zone_size = terralib.sizeof(zone)
    var point_size = terralib.sizeof(point)
    var side_size = [ terralib.sizeof(side(wild,wild,wild,wild)) ]
    c.printf("Mesh memory usage:\n")
    c.printf("  Zones  : %9lld * %4d bytes = %11lld bytes\n", conf.nz, zone_size, conf.nz * zone_size)
    c.printf("  Points : %9lld * %4d bytes = %11lld bytes\n", conf.np, point_size, conf.np * point_size)
    c.printf("  Sides  : %9lld * %4d bytes = %11lld bytes\n", conf.ns, side_size, conf.ns * side_size)
    var total = ((conf.nz * zone_size) + (conf.np * point_size) + (conf.ns * side_size))
    c.printf("  Total                             %11lld bytes\n", total)
  end

  return conf
end
end
read_config:compile()

--
-- Mesh Generator
--

struct raw_mesh_colors {
  pointcolors : &int64,
  pointmcolors : &uint64,
  pointspancolors : &int64,
  zonecolors : &int64,
  zonespancolors : &int64,
  zonesize : &int64,
}

struct mesh_colorings {
  rz_all_c : c.legion_coloring_t,
  -- rz_spans_c : c.legion_coloring_t,
  rp_all_c : c.legion_coloring_t,
  rp_all_private_c : c.legion_coloring_t,
  rp_all_ghost_c : c.legion_coloring_t,
  rp_all_shared_c : c.legion_coloring_t,
  -- rp_spans_c : c.legion_coloring_t,
  rs_all_c : c.legion_coloring_t,
  -- nspans_zones : int64,
  -- nspans_points : int64,
}

terra read_input(runtime : c.legion_runtime_t,
                 ctx : c.legion_context_t,
                 rz_physical : c.legion_physical_region_t[24],
                 rz_fields : c.legion_field_id_t[24],
                 rp_physical : c.legion_physical_region_t[17],
                 rp_fields : c.legion_field_id_t[17],
                 rs_physical : c.legion_physical_region_t[34],
                 rs_fields : c.legion_field_id_t[34],
                 conf : config)

  var color_words : c.size_t = cmath.ceil(conf.npieces/64.0)

  -- Allocate buffers for the mesh generator
  var pointpos_x_size : c.size_t = conf.np
  var pointpos_y_size : c.size_t = conf.np
  var pointcolors_size : c.size_t = conf.np
  var pointmcolors_size : c.size_t = conf.np * color_words
  var pointspancolors_size : c.size_t = conf.np
  var zonestart_size : c.size_t = conf.nz
  var zonesize_size : c.size_t = conf.nz
  var zonepoints_size : c.size_t = conf.nz * conf.maxznump
  var zonecolors_size : c.size_t = conf.nz
  var zonespancolors_size : c.size_t = conf.nz

  var pointpos_x : &double = [&double](c.malloc(pointpos_x_size*sizeof(double)))
  var pointpos_y : &double = [&double](c.malloc(pointpos_y_size*sizeof(double)))
  var pointcolors : &int64 = [&int64](c.malloc(pointcolors_size*sizeof(int64)))
  var pointmcolors : &uint64 = [&uint64](c.malloc(pointmcolors_size*sizeof(uint64)))
  var pointspancolors : &int64 = [&int64](c.malloc(pointspancolors_size*sizeof(int64)))
  var zonestart : &int64 = [&int64](c.malloc(zonestart_size*sizeof(int64)))
  var zonesize : &int64 = [&int64](c.malloc(zonesize_size*sizeof(int64)))
  var zonepoints : &int64 = [&int64](c.malloc(zonepoints_size*sizeof(int64)))
  var zonecolors : &int64 = [&int64](c.malloc(zonecolors_size*sizeof(int64)))
  var zonespancolors : &int64 = [&int64](c.malloc(zonespancolors_size*sizeof(int64)))

  regentlib.assert(pointpos_x ~= nil, "pointpos_x nil")
  regentlib.assert(pointpos_y ~= nil, "pointpos_y nil")
  regentlib.assert(pointcolors ~= nil, "pointcolors nil")
  regentlib.assert(pointmcolors ~= nil, "pointmcolors nil")
  regentlib.assert(pointspancolors ~= nil, "pointspancolors nil")
  regentlib.assert(zonestart ~= nil, "zonestart nil")
  regentlib.assert(zonesize ~= nil, "zonesize nil")
  regentlib.assert(zonepoints ~= nil, "zonepoints nil")
  regentlib.assert(zonecolors ~= nil, "zonecolors nil")
  regentlib.assert(zonespancolors ~= nil, "zonespancolors nil")

  var nspans_zones : int64 = 0
  var nspans_points : int64 = 0

  -- Call the mesh generator
  cpennant.generate_mesh_raw(
    conf.np,
    conf.nz,
    conf.nzx,
    conf.nzy,
    conf.lenx,
    conf.leny,
    conf.numpcx,
    conf.numpcy,
    conf.npieces,
    conf.meshtype,
    conf.compact,
    conf.stripsize,
    conf.spansize,
    pointpos_x, &pointpos_x_size,
    pointpos_y, &pointpos_y_size,
    pointcolors, &pointcolors_size,
    pointmcolors, &pointmcolors_size,
    pointspancolors, &pointspancolors_size,
    zonestart, &zonestart_size,
    zonesize, &zonesize_size,
    zonepoints, &zonepoints_size,
    zonecolors, &zonecolors_size,
    zonespancolors, &zonespancolors_size,
    &nspans_zones,
    &nspans_points)

  -- Write mesh data into regions
  do
    var rz_znump = c.legion_physical_region_get_field_accessor_array_1d(
      rz_physical[23], rz_fields[23])

    for i = 0, conf.nz do
      var p = c.legion_ptr_t { value = i }
      regentlib.assert(zonesize[i] < 255, "zone has more than 255 sides")
      @[&uint8](c.legion_accessor_array_1d_ref(rz_znump, p)) = uint8(zonesize[i])
    end

    c.legion_accessor_array_1d_destroy(rz_znump)
  end

  do
    var rp_px_x = c.legion_physical_region_get_field_accessor_array_1d(
      rp_physical[4], rp_fields[4])
    var rp_px_y = c.legion_physical_region_get_field_accessor_array_1d(
      rp_physical[5], rp_fields[5])
    var rp_has_bcx = c.legion_physical_region_get_field_accessor_array_1d(
      rp_physical[15], rp_fields[15])
    var rp_has_bcy = c.legion_physical_region_get_field_accessor_array_1d(
      rp_physical[16], rp_fields[16])

    var eps : double = 1e-12
    for i = 0, conf.np do
      var p = c.legion_ptr_t { value = i }
      @[&double](c.legion_accessor_array_1d_ref(rp_px_x, p)) = pointpos_x[i]
      @[&double](c.legion_accessor_array_1d_ref(rp_px_y, p)) = pointpos_y[i]

      @[&bool](c.legion_accessor_array_1d_ref(rp_has_bcx, p)) = (
        (conf.bcx_n > 0 and cmath.fabs(pointpos_x[i] - conf.bcx[0]) < eps) or
        (conf.bcx_n > 1 and cmath.fabs(pointpos_x[i] - conf.bcx[1]) < eps))
      @[&bool](c.legion_accessor_array_1d_ref(rp_has_bcy, p)) = (
        (conf.bcy_n > 0 and cmath.fabs(pointpos_y[i] - conf.bcy[0]) < eps) or
        (conf.bcy_n > 1 and cmath.fabs(pointpos_y[i] - conf.bcy[1]) < eps))
    end

    c.legion_accessor_array_1d_destroy(rp_px_x)
    c.legion_accessor_array_1d_destroy(rp_px_y)
    c.legion_accessor_array_1d_destroy(rp_has_bcx)
    c.legion_accessor_array_1d_destroy(rp_has_bcy)
  end

  do
    var rs_mapsz = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[0], rs_fields[0])
    var rs_mapsp1_ptr = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[1], rs_fields[1])
    var rs_mapsp1_index = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[2], rs_fields[2])
    var rs_mapsp2_ptr = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[3], rs_fields[3])
    var rs_mapsp2_index = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[4], rs_fields[4])
    var rs_mapss3 = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[5], rs_fields[5])
    var rs_mapss4 = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[6], rs_fields[6])

    var sstart = 0
    for iz = 0, conf.nz do
      var zsize = zonesize[iz]
      var zstart = zonestart[iz]
      for izs = 0, zsize do
        var izs3 = (izs + zsize - 1)%zsize
        var izs4 = (izs + 1)%zsize

        var p = c.legion_ptr_t { value = sstart + izs }
        @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(rs_mapsz, p)) = c.legion_ptr_t { value = iz }
        @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(rs_mapsp1_ptr, p)) = c.legion_ptr_t { value = zonepoints[zstart + izs] }
        @[&uint8](c.legion_accessor_array_1d_ref(rs_mapsp1_index, p)) = 0
        @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(rs_mapsp2_ptr, p)) = c.legion_ptr_t { value = zonepoints[zstart + izs4] }
        @[&uint8](c.legion_accessor_array_1d_ref(rs_mapsp2_index, p)) = 0
        @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(rs_mapss3, p)) = c.legion_ptr_t { value = sstart + izs3 }
        @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(rs_mapss4, p)) = c.legion_ptr_t { value = sstart + izs4 }
      end
      sstart = sstart + zsize
    end

    c.legion_accessor_array_1d_destroy(rs_mapsz)
    c.legion_accessor_array_1d_destroy(rs_mapsp1_ptr)
    c.legion_accessor_array_1d_destroy(rs_mapsp1_index)
    c.legion_accessor_array_1d_destroy(rs_mapsp2_ptr)
    c.legion_accessor_array_1d_destroy(rs_mapsp2_index)
    c.legion_accessor_array_1d_destroy(rs_mapss3)
    c.legion_accessor_array_1d_destroy(rs_mapss4)
  end

  -- Free buffers
  c.free(pointpos_x)
  c.free(pointpos_y)
  c.free(zonestart)
  --c.free(zonesize)
  c.free(zonepoints)

  -- package up the raw mesh colors for create_colorings next
  var rawcolors : raw_mesh_colors
  rawcolors.pointcolors = pointcolors
  rawcolors.pointmcolors = pointmcolors
  rawcolors.pointspancolors = pointspancolors
  rawcolors.zonecolors = zonecolors
  rawcolors.zonespancolors = zonespancolors
  rawcolors.zonesize = zonesize

  return rawcolors
end
read_input:compile()

struct cached_accessor { base : &int8; stride : int64; acc : c.legion_accessor_array_1d_t; }

terra cached_accessor:init(acc : c.legion_accessor_array_1d_t)
  self.acc = acc  -- for debug
  var ptr : c.legion_ptr_t
  ptr.value = 0
  self.base = [&int8](c.legion_accessor_array_1d_ref(acc, ptr))
  ptr.value = 1
  var basep1 : &int8
  basep1 = [&int8](c.legion_accessor_array_1d_ref(acc, ptr))
  self.stride = basep1 - self.base
  --c.printf("got %p + %zd\n", self.base, self.stride)
end

terra cached_accessor:ref(ptr : c.legion_ptr_t) : &int8
  var act_ptr : &int8 = self.base + ptr.value * self.stride
  --var chk_ptr : &int8 = [&int8](c.legion_accessor_array_1d_ref(self.acc, ptr))
  --regentlib.assert(act_ptr == chk_ptr, "ptr mismatch")
  return act_ptr
end

terra cache_accessor(acc : c.legion_accessor_array_1d_t) : cached_accessor
  var ca : cached_accessor
  ca:init(acc)
  return ca
end

terra slow_create_colorings(runtime : c.legion_runtime_t,
		       ctx : c.legion_context_t,
		       rz_physical : c.legion_physical_region_t[24],
		       rz_fields : c.legion_field_id_t[24],
		       rp_physical : c.legion_physical_region_t[17],
		       rp_fields : c.legion_field_id_t[17],
		       rs_physical : c.legion_physical_region_t[34],
		       rs_fields : c.legion_field_id_t[34],
                       rawcolors : raw_mesh_colors,
		       conf : config)

  -- Create colorings
  var result : mesh_colorings
  result.rz_all_c = c.legion_coloring_create()
  result.rp_all_c = c.legion_coloring_create()
  result.rp_all_private_c = c.legion_coloring_create()
  result.rp_all_ghost_c = c.legion_coloring_create()
  result.rp_all_shared_c = c.legion_coloring_create()
  result.rs_all_c = c.legion_coloring_create()

  var zonecolors = rawcolors.zonecolors
  var pointcolors = rawcolors.pointcolors
  var pointmcolors = rawcolors.pointmcolors

  -- color zones based on the raw zonecolors array
  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rz_all_c, i)
  end
  for i = 0, conf.nz do
    c.legion_coloring_add_point(
      result.rz_all_c, zonecolors[i], c.legion_ptr_t { value = i })
  end

  var pointfirstcolor : &int = [&int](c.malloc(conf.np*sizeof(int)))
  for i = 0, conf.np do
    pointfirstcolor[i] = -1
  end

  c.legion_coloring_ensure_color(result.rp_all_c, 0)
  c.legion_coloring_ensure_color(result.rp_all_c, 1)
  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rp_all_private_c, i)
    c.legion_coloring_ensure_color(result.rp_all_shared_c, i)
    c.legion_coloring_ensure_color(result.rp_all_ghost_c, i)
    c.legion_coloring_ensure_color(result.rs_all_c, i)
  end

  -- now iterate over the sides and:
  --  1) use the pointed-to zone's color to color the side
  --  2) use the side's color to color the point-to points - detect when more than one color exists
  do
    var rs_mapsz = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[0], rs_fields[0])
    var rs_mapsp1_ptr = c.legion_physical_region_get_field_accessor_array_1d(
      rs_physical[1], rs_fields[1])
    --var rs_mapsp2_ptr = c.legion_physical_region_get_field_accessor_array_1d(
    --  rs_physical[3], rs_fields[3])

    var crs_mapsz = cache_accessor(rs_mapsz)
    var crs_mapsp1_ptr = cache_accessor(rs_mapsp1_ptr)

    for i = 0, conf.ns do
      var ps = c.legion_ptr_t { value = i }

      --var pz = @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(rs_mapsz, ps)) 
      var pz = @[&c.legion_ptr_t](crs_mapsz:ref(ps)) 
      --var pp1 = @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(rs_mapsp1_ptr, ps)) 
      var pp1 = @[&c.legion_ptr_t](crs_mapsp1_ptr:ref(ps)) 
      --var pp2 = @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(rs_mapsp2_ptr, ps)) 

      var color = zonecolors[pz.value]
      c.legion_coloring_add_point(result.rs_all_c, color, ps)

      var oldcolor = pointfirstcolor[pp1.value]

      if oldcolor == -1 then
	-- first coloring of this point
	c.legion_coloring_add_point(result.rp_all_private_c, color, pp1)
	c.legion_coloring_add_point(result.rp_all_c, 0 --[[private]], pp1)
	pointfirstcolor[pp1.value] = color
      else
	if oldcolor ~= color then
	  -- change point to be a shared point
	  c.legion_coloring_delete_point(result.rp_all_private_c, oldcolor, pp1)
	  c.legion_coloring_add_point(result.rp_all_shared_c, oldcolor, pp1)
	  c.legion_coloring_delete_point(result.rp_all_c, 0 --[[private]], pp1)
	  c.legion_coloring_add_point(result.rp_all_c, 1 --[[shared]], pp1)
	  c.legion_coloring_add_point(result.rp_all_ghost_c, oldcolor, pp1)
	  c.legion_coloring_add_point(result.rp_all_ghost_c, color, pp1)
	end
      end
    end

    c.legion_accessor_array_1d_destroy(rs_mapsz)
    c.legion_accessor_array_1d_destroy(rs_mapsp1_ptr)
    --c.legion_accessor_array_1d_destroy(rs_mapsp2_ptr)
  end

  -- span-related stuff is not used in baseline version

  -- result.rz_spans_c = c.legion_coloring_create()
  -- result.rp_spans_c = c.legion_coloring_create()
  -- result.nspans_zones = nspans_zones
  -- result.nspans_points = nspans_points

  -- for i = 0, nspans_points do
  --   c.legion_coloring_ensure_color(result.rp_spans_c, i)
  -- end
  -- for i = 0, conf.np do
  --   c.legion_coloring_add_point(
  --     result.rp_spans_c, pointspancolors[i], c.legion_ptr_t { value = i })
  -- end

  -- for i = 0, nspans_zones do
  --   c.legion_coloring_ensure_color(result.rz_spans_c, i)
  -- end
  -- for i = 0, conf.nz do
  --   c.legion_coloring_add_point(
  --     result.rz_spans_c, zonespancolors[i], c.legion_ptr_t { value = i })
  -- end

  -- Free buffers
  c.free(pointfirstcolor)
  c.free(rawcolors.pointcolors)
  c.free(rawcolors.pointmcolors)
  c.free(rawcolors.pointspancolors)
  c.free(rawcolors.zonecolors)
  c.free(rawcolors.zonespancolors)

  return result
end

terra create_colorings(runtime : c.legion_runtime_t,
		       ctx : c.legion_context_t,
		       rz_physical : c.legion_physical_region_t[24],
		       rz_fields : c.legion_field_id_t[24],
		       rp_physical : c.legion_physical_region_t[17],
		       rp_fields : c.legion_field_id_t[17],
		       rs_physical : c.legion_physical_region_t[34],
		       rs_fields : c.legion_field_id_t[34],
                       rawcolors : raw_mesh_colors,
		       conf : config)

  -- Create colorings
  var result : mesh_colorings
  result.rz_all_c = c.legion_coloring_create()
  result.rp_all_c = c.legion_coloring_create()
  result.rp_all_private_c = c.legion_coloring_create()
  result.rp_all_ghost_c = c.legion_coloring_create()
  result.rp_all_shared_c = c.legion_coloring_create()
  result.rs_all_c = c.legion_coloring_create()

  var zonecolors = rawcolors.zonecolors
  var zonesize = rawcolors.zonesize
  var pointcolors = rawcolors.pointcolors
  var pointmcolors = rawcolors.pointmcolors

  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rz_all_c, i)
  end
  var i = 0
  while i < conf.nz do
    var j = i + 1
    while (j < conf.nz) and (zonecolors[i] == zonecolors[j]) do
      j = j + 1
    end
    c.legion_coloring_add_range(result.rz_all_c, zonecolors[i],
				c.legion_ptr_t { value = i },
				c.legion_ptr_t { value = j - 1 })
    i = j
  end
  -- for i = 0, conf.nz do
  --   c.legion_coloring_add_point(
  --     result.rz_all_c, zonecolors[i], c.legion_ptr_t { value = i })
  -- end

  -- for i = 0, nspans_zones do
  --   c.legion_coloring_ensure_color(result.rz_spans_c, i)
  -- end
  -- for i = 0, conf.nz do
  --   c.legion_coloring_add_point(
  --     result.rz_spans_c, zonespancolors[i], c.legion_ptr_t { value = i })
  -- end

  c.legion_coloring_ensure_color(result.rp_all_c, 0)
  c.legion_coloring_ensure_color(result.rp_all_c, 1)
  --[[var]] i = 0
  while i < conf.np do
    var j = i + 1
    while (j < conf.np) and (pointcolors[i] == pointcolors[j]) do
      j = j + 1
    end
    c.legion_coloring_add_range(result.rp_all_c, 
				[c.legion_color_t](pointcolors[i] == cpennant.MULTICOLOR),
				c.legion_ptr_t { value = i },
				c.legion_ptr_t { value = j - 1 })
    i = j
  end
  -- for i = 0, conf.np do
  --   c.legion_coloring_add_point(
  --     result.rp_all_c,
  --     [c.legion_color_t](pointcolors[i] == cpennant.MULTICOLOR),
  --     c.legion_ptr_t { value = i })
  -- end

  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rp_all_private_c, i)
  end
  for i = 0, conf.np do
    if pointcolors[i] ~= cpennant.MULTICOLOR then
      c.legion_coloring_add_point(
        result.rp_all_private_c,
        pointcolors[i],
        c.legion_ptr_t { value = i })
    end
  end

  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rp_all_ghost_c, i)
  end
  for i = 0, conf.np do
    for color = 0, conf.npieces do
      var word = i + color/64
      var bit = color % 64

      if (pointmcolors[word] and (1 << bit)) ~= 0 then
        c.legion_coloring_add_point(
          result.rp_all_ghost_c,
          color,
          c.legion_ptr_t { value = i })
      end
    end
  end

  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rp_all_shared_c, i)
  end
  for i = 0, conf.np do
    var done = false
    for color = 0, conf.npieces do
      var word = i + color/64
      var bit = color % 64

      if not  done and (pointmcolors[word] and (1 << bit)) ~= 0 then
        c.legion_coloring_add_point(
          result.rp_all_shared_c,
          color,
          c.legion_ptr_t { value = i })
        done = true
      end
    end
  end

  -- for i = 0, nspans_points do
  --   c.legion_coloring_ensure_color(result.rp_spans_c, i)
  -- end
  -- for i = 0, conf.np do
  --   c.legion_coloring_add_point(
  --     result.rp_spans_c, pointspancolors[i], c.legion_ptr_t { value = i })
  -- end

  for i = 0, conf.npieces do
    c.legion_coloring_ensure_color(result.rs_all_c, i)
  end
  do
    var sstart = 0
    for i = 0, conf.nz do
      var zsize = zonesize[i]

      c.legion_coloring_add_range(
        result.rs_all_c,
        zonecolors[i],
        c.legion_ptr_t { value = sstart },
        c.legion_ptr_t { value = sstart + zsize - 1 })

      sstart = sstart + zsize
    end
  end

  -- Free buffers
  c.free(rawcolors.pointcolors)
  c.free(rawcolors.pointmcolors)
  c.free(rawcolors.pointspancolors)
  c.free(rawcolors.zonecolors)
  c.free(rawcolors.zonespancolors)
  c.free(rawcolors.zonesize)

  return result
end
create_colorings:compile()

do
local solution_filename_maxlen = 1024
terra validate_output(runtime : c.legion_runtime_t,
                      ctx : c.legion_context_t,
                      rz_physical : c.legion_physical_region_t[24],
                      rz_fields : c.legion_field_id_t[24],
                      rp_physical : c.legion_physical_region_t[17],
                      rp_fields : c.legion_field_id_t[17],
                      rs_physical : c.legion_physical_region_t[34],
                      rs_fields : c.legion_field_id_t[34],
                      conf : config)
  c.printf("Running validate_output (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6)

  var solution_zr : &double = [&double](c.malloc(conf.nz*sizeof(double)))
  var solution_ze : &double = [&double](c.malloc(conf.nz*sizeof(double)))
  var solution_zp : &double = [&double](c.malloc(conf.nz*sizeof(double)))

  regentlib.assert(solution_zr ~= nil, "solution_zr nil")
  regentlib.assert(solution_ze ~= nil, "solution_ze nil")
  regentlib.assert(solution_zp ~= nil, "solution_zp nil")

  var input_filename = get_positional_arg()
  regentlib.assert(input_filename ~= nil, "input_filename nil")

  var solution_filename : int8[solution_filename_maxlen]
  do
    var sep = cstring.strrchr(input_filename, (".")[0])
    if sep == nil then
      c.printf("Error: Failed to find file extention in \"%s\"\n", input_filename)
      c.abort()
    end
    var len : int64 = [int64](sep - input_filename)
    regentlib.assert(len + 8 < solution_filename_maxlen, "solution_filename exceeds maximum length")
    cstring.strncpy(solution_filename, input_filename, len)
    cstring.strncpy(solution_filename + len, ".xy.std", 8)
  end

  c.printf("Reading \"%s\"...\n", solution_filename)
  var solution_file = c.fopen(solution_filename, "r")
  if solution_file == nil then
    c.printf("Warning: Failed to open \"%s\"\n", solution_filename)
    c.printf("Warning: Skipping validation step\n")
    return
  end

  c.fscanf(solution_file, " # zr")
  for i = 0, conf.nz do
    var iz : int64
    var zr : double
    var count = c.fscanf(
      solution_file,
      [" " .. get_type_specifier(int64, true) .. " " .. get_type_specifier(double, true)],
      &iz, &zr)
    if count ~= 2 then
      c.printf("Error: malformed file, expected 2 and got %d\n", count)
      c.abort()
    end
    solution_zr[i] = zr
  end

  c.fscanf(solution_file, " # ze")
  for i = 0, conf.nz do
    var iz : int64
    var ze : double
    var count = c.fscanf(
      solution_file,
      [" " .. get_type_specifier(int64, true) .. " " .. get_type_specifier(double, true)],
      &iz, &ze)
    if count ~= 2 then
      c.printf("Error: malformed file, expected 2 and got %d\n", count)
      c.abort()
    end
    solution_ze[i] = ze
  end

  c.fscanf(solution_file, " # zp")
  for i = 0, conf.nz do
    var iz : int64
    var zp : double
    var count = c.fscanf(
      solution_file,
      [" " .. get_type_specifier(int64, true) .. " " .. get_type_specifier(double, true)],
      &iz, &zp)
    if count ~= 2 then
      c.printf("Error: malformed file, expected 2 and got %d\n", count)
      c.abort()
    end
    solution_zp[i] = zp
  end

  var absolute_eps = 1.0e-8
  var absolute_eps_text = get_optional_arg("-absolute")
  if absolute_eps_text ~= nil then
    absolute_eps = c.atof(absolute_eps_text)
  end

  var relative_eps = 1.0e-8
  var relative_eps_text = get_optional_arg("-relative")
  if relative_eps_text ~= nil then
    relative_eps = c.atof(relative_eps_text)
  end

  -- FIXME: This is kind of silly, but some of the really small values
  -- (around 1e-17) have fairly large relative error (1e-3), tripping
  -- up the validator. For now, stop complaining about those cases if
  -- the absolute error is small.
  var relative_absolute_eps = 1.0e-17
  var relative_absolute_eps_text = get_optional_arg("-relative_absolute")
  if relative_absolute_eps_text ~= nil then
    relative_absolute_eps = c.atof(relative_absolute_eps_text)
  end

  do
    var rz_zr = c.legion_physical_region_get_field_accessor_array_1d(
      rz_physical[12], rz_fields[12])
    for i = 0, conf.nz do
      var p = c.legion_ptr_t { value = i }
      var ck = @[&double](c.legion_accessor_array_1d_ref(rz_zr, p))
      var sol = solution_zr[i]
      if cmath.fabs(ck - sol) > absolute_eps or
        (cmath.fabs(ck - sol) / sol > relative_eps and
           cmath.fabs(ck - sol) > relative_absolute_eps)
      then
        c.printf("Error: zr value out of bounds at %d, expected %.12e and got %.12e\n",
                 i, sol, ck)
        c.printf("absolute %.12e relative %.12e\n",
                 cmath.fabs(ck - sol),
                 cmath.fabs(ck - sol) / sol)
        c.abort()
      end
    end
    c.legion_accessor_array_1d_destroy(rz_zr)
  end

  do
    var rz_ze = c.legion_physical_region_get_field_accessor_array_1d(
      rz_physical[13], rz_fields[13])
    for i = 0, conf.nz do
      var p = c.legion_ptr_t { value = i }
      var ck = @[&double](c.legion_accessor_array_1d_ref(rz_ze, p))
      var sol = solution_ze[i]
      if cmath.fabs(ck - sol) > absolute_eps or
        (cmath.fabs(ck - sol) / sol > relative_eps and
           cmath.fabs(ck - sol) > relative_absolute_eps)
      then
        c.printf("Error: ze value out of bounds at %d, expected %.8e and got %.8e\n",
                 i, sol, ck)
        c.printf("absolute %.12e relative %.12e\n",
                 cmath.fabs(ck - sol),
                 cmath.fabs(ck - sol) / sol)
        c.abort()
      end
    end
    c.legion_accessor_array_1d_destroy(rz_ze)
  end

  do
    var rz_zp = c.legion_physical_region_get_field_accessor_array_1d(
      rz_physical[17], rz_fields[17])
    for i = 0, conf.nz do
      var p = c.legion_ptr_t { value = i }
      var ck = @[&double](c.legion_accessor_array_1d_ref(rz_zp, p))
      var sol = solution_zp[i]
      if cmath.fabs(ck - sol) > absolute_eps or
        (cmath.fabs(ck - sol) / sol > relative_eps and
           cmath.fabs(ck - sol) > relative_absolute_eps)
      then
        c.printf("Error: zp value out of bounds at %d, expected %.8e and got %.8e\n",
                 i, sol, ck)
        c.printf("absolute %.12e relative %.12e\n",
                 cmath.fabs(ck - sol),
                 cmath.fabs(ck - sol) / sol)
        c.abort()
      end
    end
    c.legion_accessor_array_1d_destroy(rz_zp)
  end

  c.printf("Successfully validate output\n")

  c.free(solution_zr)
  c.free(solution_ze)
  c.free(solution_zp)
end
validate_output:compile()
end

task test()
  c.printf("Running test (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6)

  var conf : config = read_config()

  var rz_all = region(ispace(ptr, conf.nz), zone)
  var rp_all = region(ispace(ptr, conf.np), point)
  var rs_all = region(ispace(ptr, conf.ns), side(wild, wild, wild, wild))

  var ts1 = c.legion_get_current_time_in_micros()
  var rawcolors = read_input(
    __runtime(), __context(),
    __physical(rz_all), __fields(rz_all),
    __physical(rp_all), __fields(rp_all),
    __physical(rs_all), __fields(rs_all),
    conf)

  var ts2 = c.legion_get_current_time_in_micros()
  var colorings = create_colorings(
    __runtime(), __context(),
    __physical(rz_all), __fields(rz_all),
    __physical(rp_all), __fields(rp_all),
    __physical(rs_all), __fields(rs_all),
    rawcolors,
    conf)

  var ts3 = c.legion_get_current_time_in_micros()
  -- Partition zones into disjoint pieces.
  var rz_all_p = partition(disjoint, rz_all, colorings.rz_all_c)

  -- Partition points into private and ghost regions.
  var rp_all_p = partition(disjoint, rp_all, colorings.rp_all_c)
  var rp_all_private = rp_all_p[0]
  var rp_all_ghost = rp_all_p[1]

  -- Partition private points into disjoint pieces by zone.
  var rp_all_private_p = partition(
    disjoint, rp_all_private, colorings.rp_all_private_c)

  -- Partition ghost points into aliased pieces by zone.
  var rp_all_ghost_p = partition(
    aliased, rp_all_ghost, colorings.rp_all_ghost_c)

  -- Partition ghost points into disjoint pieces, breaking ties
  -- between zones so that each point goes into one region only.
  var rp_all_shared_p = partition(
    disjoint, rp_all_ghost, colorings.rp_all_shared_c)

  -- Partition sides into disjoint pieces by zone.
  var rs_all_p = partition(disjoint, rs_all, colorings.rs_all_c)

  var ts4 = c.legion_get_current_time_in_micros()
  for i = 0, conf.npieces do
    var _ = init_pointers(rz_all_p[i],
			  rp_all_private_p[i],
			  rp_all_ghost_p[i],
			  rs_all_p[i])
    wait_for(_)
  end

  var ts5 = c.legion_get_current_time_in_micros()
  c.printf("partitioning time: load=%llu color=%llu part=%llu check=%llu\n",
	   ts2 - ts1, ts3 - ts2, ts4 - ts3, ts5 - ts4);

  c.printf("Initializing (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6)
  initialize(rz_all, rz_all_p,
             rp_all,
             rp_all_private, rp_all_private_p,
             rp_all_ghost, rp_all_ghost_p, rp_all_shared_p,
             rs_all, rs_all_p,
             conf)
  -- Hack: Force main task to wait for initialization to finish.
  do
    var _ = 0
    for i = 0, conf.npieces do
      _ += dummy(rz_all_p[i])
    end
    wait_for(_)
  end

  c.printf("Starting simulation (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6)
  var start_time = c.legion_get_current_time_in_micros()/1.e6
  simulate(rz_all, rz_all_p,
           rp_all,
           rp_all_private, rp_all_private_p,
           rp_all_ghost, rp_all_ghost_p, rp_all_shared_p,
           rs_all, rs_all_p,
           conf)
  -- Hack: Force main task to wait for simulation to finish.
  do
    var _ = 0
    for i = 0, conf.npieces do
      _ += dummy(rz_all_p[i])
    end
    wait_for(_)
  end
  var stop_time = c.legion_get_current_time_in_micros()/1.e6
  c.printf("Elapsed time = %.6e\n", stop_time - start_time)

  if conf.nocheck then
    c.printf("Skipping result validation due to -nocheck\n")
  else
    validate_output(
      __runtime(), __context(),
      __physical(rz_all), __fields(rz_all),
      __physical(rp_all), __fields(rp_all),
      __physical(rs_all), __fields(rs_all),
      conf)
  end

  -- write_output(conf, rz_all, rp_all, rs_all)
end

task toplevel()
  test()
end
function _t3() end -- seems like I need this to break up the statements
(terra() c.printf("Starting Legion runtime (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6) end)()
cpennant.register_mappers()
regentlib.start(toplevel)
