-- Copyright 2018 Stanford University
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

-- This file is not meant to be run directly.

-- runs-with:
-- []

import "regent"

-- Compile and link pennant.cc
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
  local pennant_cc = root_dir .. "pennant.cc"
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    pennant_so = out_dir .. "libpennant.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    pennant_so = root_dir .. "libpennant.so"
  else
    pennant_so = os.tmpname() .. ".so" -- root_dir .. "pennant.so"
  end
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
                pennant_cc .. " -o " .. pennant_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. pennant_cc)
    assert(false)
  end
  terralib.linklibrary(pennant_so)
  cpennant = terralib.includec("pennant.h", {"-I", root_dir, "-I", runtime_dir})
end

-- Also copy input files into the destination directory.
if os.getenv('OBJNAME') then
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = os.getenv('OBJNAME'):match('.*/')
  if out_dir then
    os.execute("cp -rv " .. root_dir .. "pennant.tests " .. out_dir)
  end
end

local c = regentlib.c
local cmath = terralib.includec("math.h")
local cstring = terralib.includec("string.h")

-- Hack: Make everything global for now, so it's available unqualified
-- in the caller.

sqrt = terralib.intrinsic("llvm.sqrt.f64", double -> double)

-- #####################################
-- ## Data Structures
-- #################

-- Import max/min for Terra
max = regentlib.fmax
min = regentlib.fmin

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
  {field = "par_init", type = bool, default_value = true},
  {field = "seq_init", type = bool, default_value = false},
  {field = "print_ts", type = bool, default_value = false},
  {field = "enable", type = bool, default_value = true},
  {field = "warmup", type = bool, default_value = false},
  {field = "prune", type = int64, default_value = 0},
  {field = "compact", type = bool, default_value = true},
  {field = "internal", type = bool, default_value = true},
  {field = "interior", type = bool, default_value = true},
  {field = "stripsize", type = int64, default_value = 128},
  {field = "spansize", type = int64, default_value = 2048},
  {field = "nspans_zones", type = int64, default_value = 0},
  {field = "nspans_points", type = int64, default_value = 0},
})

local config_fields_all = terralib.newlist()
config_fields_all:insertall(config_fields_input)
config_fields_all:insertall(config_fields_meshgen)
config_fields_all:insertall(config_fields_mesh)
config_fields_all:insertall(config_fields_cmd)

config = terralib.types.newstruct("config")
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

fspace span {
  start : int64,
  stop  : int64,
  internal : bool,
}

--
-- Command Line Processing
--

local terra get_positional_arg()
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

local terra get_optional_arg(key : rawstring)
  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if cstring.strcmp(args.argv[i], key) == 0 then
      if i + 1 < args.argc then
        return args.argv[i + 1]
      else
        return nil
      end
    end
    i = i + 1
  end
  return nil
end

--
-- Configuration
--

local terra get_mesh_config(conf : &config)
  -- Calculate mesh size (nzx, nzy) and dimensions (lenx, leny).
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

  -- Calculate numbers of mesh elements (nz, np and ns).
  conf.nz = conf.nzx * conf.nzy
  conf.np = (conf.nzx + 1) * (conf.nzy + 1)
  if conf.meshtype ~= MESH_HEX then
    conf.maxznump = 4
  else
    conf.maxznump = 6
  end
  conf.ns = conf.nz * conf.maxznump
end

local terra get_submesh_config(conf : &config)
  -- Calculate numbers of submeshes.
  var nx : double, ny : double = conf.nzx, conf.nzy
  var swapflag = nx > ny
  if swapflag then nx, ny = ny, nx end
  var n = sqrt(conf.npieces * nx / ny)
  var n1 : int64 = max(cmath.floor(n + 1e-12), 1)
  while conf.npieces % n1 ~= 0 do n1 = n1 - 1 end
  var n2 : int64 = cmath.ceil(n - 1e-12)
  while conf.npieces % n2 ~= 0 do n2 = n2 + 1 end
  var longside1 = max(nx / n1, ny / (conf.npieces/n1))
  var longside2 = max(nx / n2, ny / (conf.npieces/n2))
  if longside1 <= longside2 then
    conf.numpcx = n1
  else
    conf.numpcx = n2
  end
  conf.numpcy = conf.npieces / conf.numpcx
  if swapflag then conf.numpcx, conf.numpcy = conf.numpcy, conf.numpcx end
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

local function explode_array(t, value)
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

  var numpcx = get_optional_arg("-numpcx")
  if numpcx ~= nil then
    conf.numpcx = c.atoll(numpcx)
  end
  var numpcy = get_optional_arg("-numpcy")
  if numpcy ~= nil then
    conf.numpcy = c.atoll(numpcy)
  end
  if (conf.numpcx > 0 or conf.numpcy > 0) and
    conf.numpcx * conf.numpcy ~= conf.npieces
  then
    c.printf("Error: numpcx (%lld) * numpcy (%lld) must be == npieces (%lld)\n",
             conf.numpcx, conf.numpcy, conf.npieces)
    c.abort()
  end

  var par_init = get_optional_arg("-par_init")
  if par_init ~= nil then
    conf.par_init = [bool](c.atoll(par_init))
  end

  var seq_init = get_optional_arg("-seq_init")
  if seq_init ~= nil then
    conf.seq_init = [bool](c.atoll(seq_init))
  end

  var warmup = get_optional_arg("-warmup")
  if warmup ~= nil then
    conf.warmup = [bool](c.atoll(warmup))
  end

  var prune = get_optional_arg("-prune")
  if prune ~= nil then
    conf.prune = c.atoll(prune)
  end

  var compact = get_optional_arg("-compact")
  if compact ~= nil then
    conf.compact = [bool](c.atoll(compact))
  end

  var internal = get_optional_arg("-internal")
  if internal ~= nil then
    conf.internal = [bool](c.atoll(internal))
  end

  var interior = get_optional_arg("-interior")
  if interior ~= nil then
    conf.interior = [bool](c.atoll(interior))
  end

  var stripsize = get_optional_arg("-stripsize")
  if stripsize ~= nil then
    conf.stripsize = c.atoll(stripsize)
  end

  var spansize = get_optional_arg("-spansize")
  if spansize ~= nil then
    conf.spansize = c.atoll(spansize)
  end

  var print_ts = get_optional_arg("-print_ts")
  if print_ts ~= nil then
    conf.print_ts = [bool](c.atoll(print_ts))
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

  get_mesh_config(&conf)
  if conf.numpcx <= 0 or conf.numpcy <= 0 then
    get_submesh_config(&conf)
  end

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

-- #####################################
-- ## Mesh Generator
-- #################

struct mesh_colorings {
  rz_all_c : c.legion_coloring_t,
  rz_spans_c : c.legion_coloring_t,
  rp_all_c : c.legion_coloring_t,
  rp_all_private_c : c.legion_coloring_t,
  rp_all_ghost_c : c.legion_coloring_t,
  rp_all_shared_c : c.legion_coloring_t,
  rp_spans_c : c.legion_coloring_t,
  rs_all_c : c.legion_coloring_t,
  rs_spans_c : c.legion_coloring_t,
  nspans_zones : int64,
  nspans_points : int64,
}

local terra filter_none(i : int64, cs : &int64) return cs[i] end
local terra filter_ismulticolor(i : int64, cs : &int64)
  return int64(cs[i] == cpennant.MULTICOLOR)
end

local terra compute_coloring(ncolors : int64, nitems : int64,
                       coloring : c.legion_coloring_t,
                       colors : &int64, sizes : &int64,
                       filter : {int64, &int64} -> int64)
  if filter == nil then
    filter = filter_none
  end

  for i = 0, ncolors do
    c.legion_coloring_ensure_color(coloring, i)
  end
  do
    if nitems > 0 then
      var i_start = 0
      var i_end = 0
      var i_color = filter(0, colors)
      for i = 0, nitems do
        var i_size = 1
        if sizes ~= nil then
          i_size = sizes[i]
        end

        var color = filter(i, colors)
        if i_color ~= color then
          if i_color >= 0 then
            c.legion_coloring_add_range(
              coloring, i_color,
              c.legion_ptr_t { value = i_start },
              c.legion_ptr_t { value = i_end - 1 })
          end
          i_start = i_end
          i_color = color
        end
        i_end = i_end + i_size
      end
      if i_color >= 0 then
        c.legion_coloring_add_range(
          coloring, i_color,
          c.legion_ptr_t { value = i_start },
          c.legion_ptr_t { value = i_end - 1 })
      end
    end
  end
end

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

  -- Create colorings
  var result : mesh_colorings
  result.rz_all_c = c.legion_coloring_create()
  result.rz_spans_c = c.legion_coloring_create()
  result.rp_all_c = c.legion_coloring_create()
  result.rp_all_private_c = c.legion_coloring_create()
  result.rp_all_ghost_c = c.legion_coloring_create()
  result.rp_all_shared_c = c.legion_coloring_create()
  result.rp_spans_c = c.legion_coloring_create()
  result.rs_all_c = c.legion_coloring_create()
  result.rs_spans_c = c.legion_coloring_create()
  result.nspans_zones = nspans_zones
  result.nspans_points = nspans_points

  compute_coloring(
    conf.npieces, conf.nz, result.rz_all_c, zonecolors, nil, nil)
  compute_coloring(
    nspans_zones, conf.nz, result.rz_spans_c, zonespancolors, nil, nil)

  compute_coloring(
    2, conf.np, result.rp_all_c, pointcolors, nil, filter_ismulticolor)
  compute_coloring(
    conf.npieces, conf.np, result.rp_all_private_c, pointcolors, nil, nil)

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

  compute_coloring(
    nspans_points, conf.np, result.rp_spans_c, pointspancolors, nil, nil)

  compute_coloring(
    conf.npieces, conf.nz, result.rs_all_c, zonecolors, zonesize, nil)
  compute_coloring(
    nspans_zones, conf.nz, result.rs_spans_c, zonespancolors, zonesize, nil)

  -- Free buffers
  c.free(pointpos_x)
  c.free(pointpos_y)
  c.free(pointcolors)
  c.free(pointmcolors)
  c.free(pointspancolors)
  c.free(zonestart)
  c.free(zonesize)
  c.free(zonepoints)
  c.free(zonecolors)
  c.free(zonespancolors)

  return result
end
read_input:compile()

-- #####################################
-- ## Distributed Mesh Generator
-- #################

local terra ptr_t(x : int64)
  return c.legion_ptr_t { value = x }
end

-- Indexing scheme for ghost points:

local terra grid_p(conf : config)
  var npx, npy = conf.nzx + 1, conf.nzy + 1
  return npx, npy
end

local terra get_num_ghost(conf : config)
  var npx, npy = grid_p(conf)
  var num_ghost = (conf.numpcy - 1) * npx + (conf.numpcx - 1) * npy - (conf.numpcx - 1)*(conf.numpcy - 1)
  return num_ghost
end

local terra all_ghost_p(conf : config)
  var num_ghost = get_num_ghost(conf)
  var first_ghost : int64 = 0
  var last_ghost = num_ghost -- exclusive
  return first_ghost, last_ghost
end

local terra all_private_p(conf : config)
  var num_ghost = get_num_ghost(conf)
  var first_private = num_ghost
  var last_private = conf.np -- exclusive
  return first_private, last_private
end

local terra block_zx(conf : config, pcx : int64)
  var first_zx = conf.nzx * pcx / conf.numpcx
  var last_zx = conf.nzx * (pcx + 1) / conf.numpcx -- exclusive
  var stride_zx = last_zx - first_zx
  return first_zx, last_zx, stride_zx
end

local terra block_zy(conf : config, pcy : int64)
  var first_zy = conf.nzy * pcy / conf.numpcy
  var last_zy = conf.nzy * (pcy + 1) / conf.numpcy -- exclusive
  var stride_zy = last_zy - first_zy
  return first_zy, last_zy, stride_zy
end

local terra block_z(conf : config, pcx : int64, pcy : int64)
  var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
  var first_zy, last_zy, stride_zy = block_zy(conf, pcy)
  var first_z = first_zy * conf.nzx + first_zx * stride_zy
  var last_z = first_z + stride_zy*stride_zx -- exclusive
  return first_z, last_z
end

local terra block_px(conf : config, pcx : int64)
  var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
  var first_px = first_zx - pcx + [int64](pcx ~= 0)
  var last_px = last_zx - pcx + [int64](pcx == conf.numpcx - 1) -- exclusive
  var stride_px = last_px - first_px
  return first_px, last_px, stride_px
end

local terra block_py(conf : config, pcy : int64)
  var first_zy, last_zy, stride_zy = block_zy(conf, pcy)
  var first_py = first_zy - pcy + [int64](pcy ~= 0)
  var last_py = last_zy - pcy + [int64](pcy == conf.numpcy - 1) -- exclusive
  var stride_py = last_py - first_py
  return first_py, last_py, stride_py
end

local terra block_p(conf : config, pcx : int64, pcy : int64)
  var npx, npy = grid_p(conf)
  var first_private, last_private = all_private_p(conf)
  var first_py, last_py, stride_py = block_py(conf, pcy)
  var first_px, last_px, stride_px = block_px(conf, pcx)
  var first_p = first_private + first_py * (npx - conf.numpcx + 1) + first_px * stride_py
  var last_p = first_p + stride_py*stride_px -- exclusive
  return first_p, last_p
end

-- Ghost nodes are counted starting at the right face and moving down
-- to the bottom and then bottom-right. This is identical to a point
-- numbering where points are sorted first by number of colors (ghosts
-- first) and then by first color.
local terra ghost_first_p(conf : config, pcx : int64, pcy : int64)
  var npx, npy = conf.nzx + 1, conf.nzy + 1

  var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
  var first_zy, last_zy, stride_zy = block_zy(conf, pcy)

  -- Count previous vertical segments
  var prev_vertical_rows = (conf.numpcx - 1) * (first_zy - pcy + [int64](pcy > 0))
  var prev_vertical_row = pcx * (stride_zy - [int64](pcy > 0 and pcy < conf.numpcy - 1))

  -- Count previous horizontal segments
  var prev_horizontal_rows = pcy * npx
  var prev_horizontal_row = [int64](pcy < conf.numpcy - 1) * (first_zx + [int64](pcx > 0))

  return prev_vertical_rows + prev_vertical_row + prev_horizontal_rows + prev_horizontal_row
end

-- Corners:
local terra ghost_bottom_right_p(conf : config, pcx : int64, pcy : int64)
  if pcx < conf.numpcx - 1 and pcy < conf.numpcy - 1 then
    var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
    var first_zy, last_zy, stride_zy = block_zy(conf, pcy)

    var first_p = ghost_first_p(conf, pcx, pcy)
    var corner_p = first_p + stride_zy + stride_zx - [int64](pcy > 0) - [int64](pcx > 0)
    return corner_p, corner_p + 1
  end
  return 0, 0
end

local terra ghost_top_left_p(conf : config, pcx : int64, pcy : int64)
  if pcx > 0 and pcy > 0 then
    return ghost_bottom_right_p(conf, pcx-1, pcy-1)
  end
  return 0, 0
end


local terra ghost_top_right_p(conf : config, pcx : int64, pcy : int64)
  if pcy > 0 then
    return ghost_bottom_right_p(conf, pcx, pcy-1)
  end
  return 0, 0
end

local terra ghost_bottom_left_p(conf : config, pcx : int64, pcy : int64)
  if pcx > 0 then
    return ghost_bottom_right_p(conf, pcx-1, pcy)
  end
  return 0, 0
end

-- Faces:
local terra ghost_bottom_p(conf : config, pcx : int64, pcy : int64)
  if pcy < conf.numpcy - 1 then
    var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
    var first_zy, last_zy, stride_zy = block_zy(conf, pcy)

    var first_p = ghost_first_p(conf, pcx, pcy)
    var first_face_p = first_p + [int64](pcx < conf.numpcx - 1) * (stride_zy - [int64](pcy > 0))
    var last_face_p = first_face_p + stride_zx - 1 + [int64](pcx == 0) + [int64](pcx == conf.numpcx - 1) -- exclusive
    return first_face_p, last_face_p
  end
  return 0, 0
end

local terra ghost_top_p(conf : config, pcx : int64, pcy : int64)
  if pcy > 0 then
    return ghost_bottom_p(conf, pcx, pcy-1)
  end
  return 0, 0
end

local terra ghost_right_p(conf : config, pcx : int64, pcy : int64)
  if pcx < conf.numpcx - 1 then
    var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
    var first_zy, last_zy, stride_zy = block_zy(conf, pcy)

    var first_p = ghost_first_p(conf, pcx, pcy)
    var first_face_p = first_p
    var last_face_p = first_face_p + stride_zy - 1 + [int64](pcy == 0) + [int64](pcy == conf.numpcy - 1) -- exclusive
    return first_face_p, last_face_p
  end
  return 0, 0
end

local terra ghost_left_p(conf : config, pcx : int64, pcy : int64)
  if pcx > 0 then
    return ghost_right_p(conf, pcx-1, pcy)
  end
  return 0, 0
end

terra read_partitions(conf : config) : mesh_colorings
  regentlib.assert(conf.npieces > 0, "npieces must be > 0")
  regentlib.assert(conf.compact, "parallel initialization requires compact")
  regentlib.assert(
    conf.meshtype == MESH_RECT,
    "parallel initialization only works on rectangular meshes")
  var znump = 4

  -- Create colorings.
  var result : mesh_colorings
  result.rz_all_c = c.legion_coloring_create()
  result.rz_spans_c = c.legion_coloring_create()
  result.rp_all_c = c.legion_coloring_create()
  result.rp_all_private_c = c.legion_coloring_create()
  result.rp_all_ghost_c = c.legion_coloring_create()
  result.rp_all_shared_c = c.legion_coloring_create()
  result.rp_spans_c = c.legion_coloring_create()
  result.rs_all_c = c.legion_coloring_create()
  result.rs_spans_c = c.legion_coloring_create()

  -- Zones and sides: private partitions.
  var max_stride_zx = (conf.nzx + conf.numpcx - 1) / conf.numpcx
  var max_stride_zy = (conf.nzy + conf.numpcy - 1) / conf.numpcy
  var zspansize = conf.spansize/znump
  result.nspans_zones = (max_stride_zx*max_stride_zy + zspansize - 1) / zspansize

  for pcy = 0, conf.numpcy do
    for pcx = 0, conf.numpcx do
      var piece = pcy * conf.numpcx + pcx
      var first_z, last_z = block_z(conf, pcx, pcy)

      c.legion_coloring_add_range(
        result.rz_all_c, piece,
        ptr_t(first_z), ptr_t(last_z - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rs_all_c, piece,
        ptr_t(first_z * znump), ptr_t(last_z * znump - 1)) -- inclusive

      var span = 0
      for z = first_z, last_z, zspansize do
        c.legion_coloring_add_range(
          result.rz_spans_c, span,
          ptr_t(z), ptr_t(min(z + zspansize, last_z) - 1)) -- inclusive
        c.legion_coloring_add_range(
          result.rs_spans_c, span,
          ptr_t(z * znump), ptr_t(min(z + zspansize, last_z) * znump - 1)) -- inclusive
        span = span + 1
      end
      regentlib.assert(span <= result.nspans_zones, "zone span overflow")
    end
  end

  -- Points: top level partition (private vs ghost).
  var first_ghost, last_ghost = all_ghost_p(conf)
  var first_private, last_private = all_private_p(conf)

  var all_private_color, all_ghost_color = 0, 1
  c.legion_coloring_add_range(
    result.rp_all_c, all_ghost_color,
    ptr_t(first_ghost), ptr_t(last_ghost - 1)) -- inclusive
  c.legion_coloring_add_range(
    result.rp_all_c, all_private_color,
    ptr_t(first_private), ptr_t(last_private - 1)) -- inclusive

  -- This math is hard, so just keep track of these as we go along.
  result.nspans_points = 0

  -- Points: private partition.
  for pcy = 0, conf.numpcy do
    for pcx = 0, conf.numpcx do
      var piece = pcy * conf.numpcx + pcx
      var first_p, last_p = block_p(conf, pcx, pcy)

      c.legion_coloring_add_range(
        result.rp_all_private_c, piece,
        ptr_t(first_p), ptr_t(last_p - 1)) -- inclusive

      var span = 0
      for p = first_p, last_p, conf.spansize do
        c.legion_coloring_add_range(
          result.rp_spans_c, span,
          ptr_t(p), ptr_t(min(p + conf.spansize, last_p) - 1)) -- inclusive
        span = span + 1
      end
      result.nspans_points = max(result.nspans_points, span)
    end
  end

  -- Points: ghost and shared partitions.
  for pcy = 0, conf.numpcy do
    for pcx = 0, conf.numpcx do
      var piece = pcy * conf.numpcx + pcx

      var top_left = ghost_top_left_p(conf, pcx, pcy)
      var top_right = ghost_top_right_p(conf, pcx, pcy)
      var bottom_left = ghost_bottom_left_p(conf, pcx, pcy)
      var bottom_right = ghost_bottom_right_p(conf, pcx, pcy)
      var top = ghost_top_p(conf, pcx, pcy)
      var bottom = ghost_bottom_p(conf, pcx, pcy)
      var left = ghost_left_p(conf, pcx, pcy)
      var right = ghost_right_p(conf, pcx, pcy)

      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(top_left._0), ptr_t(top_left._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(top._0), ptr_t(top._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(top_right._0), ptr_t(top_right._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(left._0), ptr_t(left._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(right._0), ptr_t(right._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(bottom_left._0), ptr_t(bottom_left._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(bottom._0), ptr_t(bottom._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_ghost_c, piece, ptr_t(bottom_right._0), ptr_t(bottom_right._1 - 1)) -- inclusive

      c.legion_coloring_add_range(
        result.rp_all_shared_c, piece, ptr_t(right._0), ptr_t(right._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_shared_c, piece, ptr_t(bottom._0), ptr_t(bottom._1 - 1)) -- inclusive
      c.legion_coloring_add_range(
        result.rp_all_shared_c, piece, ptr_t(bottom_right._0), ptr_t(bottom_right._1 - 1)) -- inclusive

      var first_p : int64, last_p : int64 = -1, -1
      if right._0 < right._1 then
        first_p = right._0
        last_p = right._1
      end
      if bottom._0 < bottom._1 then
        if first_p < 0 then first_p = bottom._0 end
        last_p = bottom._1
      end
      if bottom_right._0 < bottom_right._1 then
        if first_p < 0 then first_p = bottom_right._0 end
        last_p = bottom_right._1
      end

      var span = 0
      for p = first_p, last_p, conf.spansize do
        c.legion_coloring_add_range(
          result.rp_spans_c, span,
          ptr_t(p), ptr_t(min(p + conf.spansize, last_p) - 1)) -- inclusive
        span = span + 1
      end
      result.nspans_points = max(result.nspans_points, span)
    end
  end

  return result
end
read_partitions:compile()

local terra get_zone_position(conf : config, pcx : int64, pcy : int64, z : int64)
  var first_zx, last_zx, stride_zx = block_zx(conf, pcx)
  var first_zy, last_zy, stride_zy = block_zy(conf, pcy)
  var first_z, last_z = block_z(conf, pcx, pcy)

  var zstripsize = conf.stripsize
  if zstripsize <= 0 then
    zstripsize = stride_zx
  end

  var z_strip_num = (z - first_z) / (zstripsize * stride_zy)
  var z_strip_elt = (z - first_z) % (zstripsize * stride_zy)
  var leftover = stride_zx - zstripsize * z_strip_num
  if leftover == 0 then leftover = zstripsize end
  var z_strip_width = min(zstripsize, leftover)
  regentlib.assert(z_strip_width > 0, "zero width strip")
  var z_strip_x = z_strip_elt % z_strip_width
  var z_strip_y = z_strip_elt / z_strip_width
  var z_x = z_strip_num * zstripsize + z_strip_x
  var z_y = z_strip_y
  return z_x, z_y
end

task initialize_spans(conf : config,
                      piece : int64,
                      rz_spans : region(span),
                      rp_spans_private : region(span),
                      rp_spans_shared : region(span),
                      rs_spans : region(span))
where
  reads writes(rz_spans, rp_spans_private, rp_spans_shared, rs_spans),
  rz_spans * rp_spans_private, rz_spans * rp_spans_shared, rz_spans * rs_spans,
  rp_spans_private * rp_spans_shared, rp_spans_private * rs_spans,
  rp_spans_shared * rs_spans
do
  -- Unfortunately, this duplicates a lot of functionality in read_partitions.

  regentlib.assert(conf.compact, "parallel initialization requires compact")
  regentlib.assert(
    conf.meshtype == MESH_RECT,
    "parallel initialization only works on rectangular meshes")
  var znump = 4
  var zspansize = conf.spansize/znump

  var pcx, pcy = piece % conf.numpcx, piece / conf.numpcx

  -- Zones and sides.
  do
    var { first_zx = _0, last_zx = _1, stride_zx = _2 } = block_zx(conf, pcx)
    var { first_zy = _0, last_zy = _1, stride_zy = _2 } = block_zy(conf, pcy)
    var { first_z = _0, last_z = _1} = block_z(conf, pcx, pcy)

    var num_external = 0
    var span_i = 0
    for z = first_z, last_z, zspansize do
      var external = true
      if conf.internal then
        if conf.interior then
          var interior_zx = stride_zx - [int64](pcx > 0) - [int64](pcx < conf.numpcx - 1)
          var interior_zy = stride_zy - [int64](pcy > 0) - [int64](pcy < conf.numpcy - 1)
          var interior = interior_zx * interior_zy
          external = min(z + zspansize, last_z) - first_z > interior
        else
          var { z0_x = _0, z0_y = _1 } = get_zone_position(conf, pcx, pcy, z)
          var { zn_x = _0, zn_y = _1 } = get_zone_position(conf, pcx, pcy, min(z + zspansize, last_z) - 1)

          var external = (zn_y > z0_y and conf.numpcx > 1) or
            (z0_x == 0 and pcx > 0) or (z0_x == stride_zx - 1 and pcx < conf.numpcx - 1) or
            (zn_x == 0 and pcx > 0) or (zn_x == stride_zx - 1 and pcx < conf.numpcx - 1) or
            (z0_y == 0 and pcy > 0) or (z0_y == stride_zy - 1 and pcy < conf.numpcy - 1) or
            (zn_y == 0 and pcy > 0) or (zn_y == stride_zy - 1 and pcy < conf.numpcy - 1)
        end
      end
      num_external += int(external)

      var zs = unsafe_cast(ptr(span, rz_spans), piece * conf.nspans_zones + span_i)
      var ss = unsafe_cast(ptr(span, rs_spans), piece * conf.nspans_zones + span_i)

      var z_span = {
        start = z,
        stop = min(z + zspansize, last_z), -- exclusive
        internal = not external,
      }
      var s_span = {
        start = z * znump,
        stop = min(z + zspansize, last_z) * znump, -- exclusive
        internal = not external,
      }
      if conf.seq_init then regentlib.assert(zs.start == z_span.start and zs.stop == z_span.stop, "bad value: zone span") end
      if conf.seq_init then regentlib.assert(ss.start == s_span.start and ss.stop == s_span.stop, "bad value: side span") end

      @zs = z_span
      @ss = s_span
      span_i = span_i + 1
    end
    regentlib.assert(span_i <= conf.nspans_zones, "zone span overflow")
    c.printf("Spans: total %ld external %ld percent external %f\n",
             span_i, num_external, double(num_external*100)/span_i)
  end

  -- Points: private spans.
  do
    var piece = pcy * conf.numpcx + pcx
    var { first_p = _0, last_p = _1 } = block_p(conf, pcx, pcy)

    var span_i = 0
    for p = first_p, last_p, conf.spansize do
      var ps = unsafe_cast(ptr(span, rp_spans_private), piece * conf.nspans_points + span_i)

      var p_span = { start = p, stop = min(p + conf.spansize, last_p), internal = false } -- exclusive
      if conf.seq_init then regentlib.assert(ps.start == p_span.start and ps.stop == p_span.stop, "bad value: private point span") end

      @ps = p_span

      span_i = span_i + 1
    end
  end

  -- Points: shared spans.
  do
    var piece = pcy * conf.numpcx + pcx

    var right = ghost_right_p(conf, pcx, pcy)
    var bottom = ghost_bottom_p(conf, pcx, pcy)
    var bottom_right = ghost_bottom_right_p(conf, pcx, pcy)

    var first_p, last_p = -1, -1
    if right._0 < right._1 then
      first_p = right._0
      last_p = right._1
    end
    if bottom._0 < bottom._1 then
      if first_p < 0 then first_p = bottom._0 end
      last_p = bottom._1
    end
    if bottom_right._0 < bottom_right._1 then
      if first_p < 0 then first_p = bottom_right._0 end
      last_p = bottom_right._1
    end

    var span_i = 0
    for p = first_p, last_p, conf.spansize do
      var ps = unsafe_cast(ptr(span, rp_spans_shared), piece * conf.nspans_points + span_i)

      var p_span = { start = p, stop = min(p + conf.spansize, last_p), internal = false } -- exclusive
      if conf.seq_init then regentlib.assert(ps.start == p_span.start and ps.stop == p_span.stop, "bad value: private point span") end

      @ps = p_span

      span_i = span_i + 1
    end
  end
end

task initialize_topology(conf : config,
                         piece : int64,
                         rz : region(zone),
                         rpp : region(point),
                         rps : region(point),
                         rpg : region(point),
                         rs : region(side(rz, rpp, rpg, rs)))
where reads writes(rz.znump,
                   rpp.{px, has_bcx, has_bcy},
                   rps.{px, has_bcx, has_bcy},
                   rs.{mapsz, mapsp1, mapsp2, mapss3, mapss4}),
  reads(rpg.{px0}) -- Hack: Work around runtime bug with no-acccess regions.
do
  regentlib.assert(
    conf.meshtype == MESH_RECT,
    "distributed initialization only works on rectangular meshes")
  var znump = 4

  var pcx, pcy = piece % conf.numpcx, piece / conf.numpcx

  -- Initialize zones.
  fill(rz.znump, znump)

  -- Initialize points: private.
  var dx = conf.lenx / double(conf.nzx)
  var dy = conf.leny / double(conf.nzy)
  var eps = 1e-12
  do
    var {first_zx = _0, last_zx = _1, stride_zx = _2} = block_zx(conf, pcx)
    var {first_zy = _0, last_zy = _1, stride_zy = _2} = block_zy(conf, pcy)
    var {first_p = _0, last_p = _1} = block_p(conf, pcx, pcy)

    var p_ = first_p
    for y = first_zy + [int64](pcy > 0), last_zy + [int64](pcy == conf.numpcy - 1) do
      for x = first_zx + [int64](pcx > 0), last_zx + [int64](pcx == conf.numpcx - 1) do
        var p = dynamic_cast(ptr(point, rpp), [ptr](p_))
        regentlib.assert(not isnull(p), "bad pointer")

        var px = { x = dx*x, y = dy*y }
        if conf.seq_init then regentlib.assert(abs(px.x - p.px.x) < eps, "bad value: px.x") end
        if conf.seq_init then regentlib.assert(abs(px.y - p.px.y) < eps, "bad value: px.y") end

        var has_bcx = (conf.bcx_n > 0 and cmath.fabs(px.x - conf.bcx[0]) < eps) or
          (conf.bcx_n > 1 and cmath.fabs(px.x - conf.bcx[1]) < eps)
        var has_bcy = (conf.bcy_n > 0 and cmath.fabs(px.y - conf.bcy[0]) < eps) or
          (conf.bcy_n > 1 and cmath.fabs(px.y - conf.bcy[1]) < eps)
        if conf.seq_init then regentlib.assert(has_bcx == p.has_bcx, "bad value: has_bcx") end
        if conf.seq_init then regentlib.assert(has_bcy == p.has_bcy, "bad value: has_bcy") end

        p.px = px
        p.has_bcx = has_bcx
        p.has_bcy = has_bcy

        p_ += 1
      end
    end
    regentlib.assert(p_ == last_p, "point underflow")
  end

  -- Initialize points: shared.
  do
    var {first_zx = _0, last_zx = _1, stride_zx = _2} = block_zx(conf, pcx)
    var {first_zy = _0, last_zy = _1, stride_zy = _2} = block_zy(conf, pcy)

    var right = ghost_right_p(conf, pcx, pcy)
    var bottom = ghost_bottom_p(conf, pcx, pcy)
    var bottom_right = ghost_bottom_right_p(conf, pcx, pcy)

    for p_ = right._0, right._1 do
      var p = dynamic_cast(ptr(point, rps), [ptr](p_))
      regentlib.assert(not isnull(p), "bad pointer")

      var x, y = last_zx, first_zy + (p_ - right._0 + [int64](pcy > 0))

      var px = { x = dx*x, y = dy*y }
      if conf.seq_init then regentlib.assert(abs(px.x - p.px.x) < eps, "bad value: px.x") end
      if conf.seq_init then regentlib.assert(abs(px.y - p.px.y) < eps, "bad value: px.y") end

      var has_bcx = (conf.bcx_n > 0 and cmath.fabs(px.x - conf.bcx[0]) < eps) or
        (conf.bcx_n > 1 and cmath.fabs(px.x - conf.bcx[1]) < eps)
      var has_bcy = (conf.bcy_n > 0 and cmath.fabs(px.y - conf.bcy[0]) < eps) or
        (conf.bcy_n > 1 and cmath.fabs(px.y - conf.bcy[1]) < eps)
      if conf.seq_init then regentlib.assert(has_bcx == p.has_bcx, "bad value: has_bcx") end
      if conf.seq_init then regentlib.assert(has_bcy == p.has_bcy, "bad value: has_bcy") end

      p.px = px
      p.has_bcx = has_bcx
      p.has_bcy = has_bcy
    end

    for p_ = bottom._0, bottom._1 do
      var p = dynamic_cast(ptr(point, rps), [ptr](p_))
      regentlib.assert(not isnull(p), "bad pointer")

      var x, y = first_zx + (p_ - bottom._0 + [int64](pcx > 0)), last_zy

      var px = { x = dx*x, y = dy*y }
      if conf.seq_init then regentlib.assert(abs(px.x - p.px.x) < eps, "bad value: px.x") end
      if conf.seq_init then regentlib.assert(abs(px.y - p.px.y) < eps, "bad value: px.y") end

      var has_bcx = (conf.bcx_n > 0 and cmath.fabs(px.x - conf.bcx[0]) < eps) or
        (conf.bcx_n > 1 and cmath.fabs(px.x - conf.bcx[1]) < eps)
      var has_bcy = (conf.bcy_n > 0 and cmath.fabs(px.y - conf.bcy[0]) < eps) or
        (conf.bcy_n > 1 and cmath.fabs(px.y - conf.bcy[1]) < eps)
      if conf.seq_init then regentlib.assert(has_bcx == p.has_bcx, "bad value: has_bcx") end
      if conf.seq_init then regentlib.assert(has_bcy == p.has_bcy, "bad value: has_bcy") end

      p.px = px
      p.has_bcx = has_bcx
      p.has_bcy = has_bcy
    end

    for p_ = bottom_right._0, bottom_right._1 do
      var p = dynamic_cast(ptr(point, rps), [ptr](p_))
      regentlib.assert(not isnull(p), "bad pointer")

      var x, y = last_zx, last_zy

      var px = { x = dx*x, y = dy*y }
      if conf.seq_init then regentlib.assert(abs(px.x - p.px.x) < eps, "bad value: px.x") end
      if conf.seq_init then regentlib.assert(abs(px.y - p.px.y) < eps, "bad value: px.y") end

      var has_bcx = (conf.bcx_n > 0 and cmath.fabs(px.x - conf.bcx[0]) < eps) or
        (conf.bcx_n > 1 and cmath.fabs(px.x - conf.bcx[1]) < eps)
      var has_bcy = (conf.bcy_n > 0 and cmath.fabs(px.y - conf.bcy[0]) < eps) or
        (conf.bcy_n > 1 and cmath.fabs(px.y - conf.bcy[1]) < eps)
      if conf.seq_init then regentlib.assert(has_bcx == p.has_bcx, "bad value: has_bcx") end
      if conf.seq_init then regentlib.assert(has_bcy == p.has_bcy, "bad value: has_bcy") end

      p.px = px
      p.has_bcx = has_bcx
      p.has_bcy = has_bcy
    end
  end

  -- Initialize sides.
  do
    var {first_zx = _0, last_zx = _1, stride_zx = _2} = block_zx(conf, pcx)
    var {first_zy = _0, last_zy = _1, stride_zy = _2} = block_zy(conf, pcy)
    var {first_z = _0, last_z = _1} = block_z(conf, pcx, pcy)
    var {first_p = _0, last_p = _1} = block_p(conf, pcx, pcy)

    var zstripsize = conf.stripsize
    if zstripsize <= 0 then
      zstripsize = stride_zx
    end

    var z_ = first_z
    var passes = 1 + [int64](conf.interior)
    for pass = 0, passes do
      for x0 = 0, stride_zx, conf.stripsize do
        for y = 0, stride_zy do
          for x = x0, min(x0 + conf.stripsize, stride_zx) do
            if not conf.interior or
              (pass == 0) ~=
                 ((y == 0 and pcy > 0) or (x == 0 and pcx > 0) or
                  (y == stride_zy - 1 and pcy < conf.numpcy - 1) or
                  (x == stride_zx - 1 and pcx < conf.numpcx - 1))
            then
              var z = dynamic_cast(ptr(zone, rz), [ptr](z_))
              regentlib.assert(not isnull(z), "bad pointer")

              var top_left = ghost_top_left_p(conf, pcx, pcy)
              var top_right = ghost_top_right_p(conf, pcx, pcy)
              var bottom_left = ghost_bottom_left_p(conf, pcx, pcy)
              var bottom_right = ghost_bottom_right_p(conf, pcx, pcy)
              var top = ghost_top_p(conf, pcx, pcy)
              var bottom = ghost_bottom_p(conf, pcx, pcy)
              var left = ghost_left_p(conf, pcx, pcy)
              var right = ghost_right_p(conf, pcx, pcy)

              var inner_x = x - [int64](pcx > 0)
              var inner_y = y - [int64](pcy > 0)
              var inner_stride_x = stride_zx - 1 + [int64](pcx == 0) + [int64](pcx == conf.numpcx - 1)
              var pp : ptr(point, rpp, rpg)[4]
              do
                var p_ : int64 = -1
                if y == 0 and x == 0 and pcy > 0 and pcx > 0 then
                  p_ = top_left._0
                elseif y == 0 and pcy > 0 then
                  p_ = top._0 + inner_x
                elseif x == 0 and pcx > 0 then
                  p_ = left._0 + inner_y
                else -- private
                  p_ = first_p + inner_y * inner_stride_x + inner_x
                end
                var p = dynamic_cast(ptr(point, rpp, rpg), [ptr](p_))
                regentlib.assert(not isnull(p), "bad pointer")
                pp[0] = p
              end
              do
                var p_ : int64 = -1
                if y == 0 and x == stride_zx - 1 and pcy > 0 and pcx < conf.numpcx - 1 then
                  p_ = top_right._0
                elseif y == 0 and pcy > 0 then
                  p_ = top._0 + (inner_x + 1)
                elseif x == stride_zx - 1 and pcx < conf.numpcx - 1 then
                  p_ = right._0 + inner_y
                else -- private
                  p_ = first_p + inner_y * inner_stride_x + (inner_x + 1)
                end
                var p = dynamic_cast(ptr(point, rpp, rpg), [ptr](p_))
                regentlib.assert(not isnull(p), "bad pointer")
                pp[1] = p
              end
              do
                var p_ : int64 = -1
                if y == stride_zy - 1 and x == stride_zx - 1 and pcy < conf.numpcy - 1 and pcx < conf.numpcx - 1 then
                  p_ = bottom_right._0
                elseif y == stride_zy - 1 and pcy < conf.numpcy - 1 then
                  p_ = bottom._0 + (inner_x + 1)
                elseif x == stride_zx - 1 and pcx < conf.numpcx - 1 then
                  p_ = right._0 + (inner_y + 1)
                else -- private
                  p_ = first_p + (inner_y + 1) * inner_stride_x + (inner_x + 1)
                end
                var p = dynamic_cast(ptr(point, rpp, rpg), [ptr](p_))
                regentlib.assert(not isnull(p), "bad pointer")
                pp[2] = p
              end
              do
                var p_ : int64 = -1
                if y == stride_zy - 1 and x == 0 and pcy < conf.numpcy - 1 and pcx > 0 then
                  p_ = bottom_left._0
                elseif y == stride_zy - 1 and pcy < conf.numpcy - 1 then
                  p_ = bottom._0 + inner_x
                elseif x == 0 and pcx > 0 then
                  p_ = left._0 + (inner_y + 1)
                else -- private
                  p_ = first_p + (inner_y + 1) * inner_stride_x + inner_x
                end
                var p = dynamic_cast(ptr(point, rpp, rpg), [ptr](p_))
                regentlib.assert(not isnull(p), "bad pointer")
                pp[3] = p
              end

              var ss : ptr(side(rz, rpp, rpg, rs), rs)[4]
              for i = 0, znump do
                var s_ = z_ * znump + i
                var s = dynamic_cast(ptr(side(rz, rpp, rpg, rs), rs), [ptr](s_))
                regentlib.assert(not isnull(s), "bad pointer")
                ss[i] = s
              end

              for i = 0, znump do
                var prev_i = (i + znump - 1) % znump
                var next_i = (i + 1) % znump

                var s = ss[i]
                if conf.seq_init then regentlib.assert(s.mapsz == z, "bad value: mapsz") end
                s.mapsz = z

                if conf.seq_init then regentlib.assert(s.mapsp1 == pp[i], "bad value: mapsp1") end
                if conf.seq_init then regentlib.assert(s.mapsp2 == pp[next_i], "bad value: mapsp2") end
                s.mapsp1 = pp[i]
                s.mapsp2 = pp[next_i]

                if conf.seq_init then regentlib.assert(s.mapss3 == ss[prev_i], "bad value: mapss3") end
                if conf.seq_init then regentlib.assert(s.mapss4 == ss[next_i], "bad value: mapss4") end
                s.mapss3 = ss[prev_i]
                s.mapss4 = ss[next_i]
              end

              z_ += 1
            end
          end
        end
      end
    end
    regentlib.assert(z_ == last_z, "zone underflow")
  end
end

--
-- Validation
--

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

local common = {}
return common
