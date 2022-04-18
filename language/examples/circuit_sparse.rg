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
-- [
--   ["-ll:cpu", "4"],
--   ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2", "-ftrace", "0", "-freplicable", "0"],
--   ["-ll:cpu", "4", "-dm:memoize", "-ffuture", "0"],
--   ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2", "-dm:memoize", "-freplicable", "0"],
--   ["-ll:cpu", "5", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "5", "-p", "5", "-freplicable", "0"]
-- ]

import "regent"

local format = require("std/format")

local use_python_main = rawget(_G, "circuit_use_python_main") == true

-- Compile and link circuit_mapper.cc
local cmapper
local cconfig
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

  local mapper_cc = root_dir .. "circuit_mapper.cc"
  local mapper_so
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    mapper_so = out_dir .. "libcircuit_mapper.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    mapper_so = root_dir .. "libcircuit_mapper.so"
  else
    mapper_so = os.tmpname() .. ".so" -- root_dir .. "circuit_mapper.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
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
  cmapper = terralib.includec("circuit_mapper.h", include_dirs)
  cconfig = terralib.includec("circuit_config.h", include_dirs)
end

local c = regentlib.c
local std = terralib.includec("stdlib.h")
local cmath = terralib.includec("math.h")
local cstring = terralib.includec("string.h")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)
rawset(_G, "ceil", cmath.ceil)

WIRE_SEGMENTS = 10
STEPS = 10000
DELTAT = 1e-6

struct Colorings {
  privacy_map : c.legion_point_coloring_t,
  private_node_map : c.legion_point_coloring_t,
  shared_node_map : c.legion_point_coloring_t,
}

local Config = cconfig.Config

fspace node {
  node_cap : float,
  leakage  : float,
  charge   : float,
  node_voltage  : float,
}

struct currents {
  _0 : float,
  _1 : float,
  _2 : float,
  _3 : float,
  _4 : float,
  _5 : float,
  _6 : float,
  _7 : float,
  _8 : float,
  _9 : float,
}

struct voltages {
  _0 : float,
  _1 : float,
  _2 : float,
  _3 : float,
  _4 : float,
  _5 : float,
  _6 : float,
  _7 : float,
  _8 : float,
}

struct ghost_range {
  rect : rect1d,
}

fspace wire(rpn : region(node),
            rsn : region(node),
            rgn : region(node)) {
  in_ptr : ptr(node, rpn, rsn),
  out_ptr : ptr(node, rpn, rsn, rgn),
  inductance : float,
  resistance : float,
  wire_cap : float,
  current : currents,
  voltage : voltages,
}

fspace timestamp {
  init_start : int64,
  init_stop : int64,
  start : int64,
  stop : int64,
}

if not use_python_main then

terra parse_input_args(conf : Config)
  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstring.strcmp(args.argv[i], "-l") == 0 then
      i = i + 1
      conf.num_loops = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-i") == 0 then
      i = i + 1
      conf.steps = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-p") == 0 then
      i = i + 1
      conf.num_pieces = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-pps") == 0 then
      i = i + 1
      conf.pieces_per_superpiece = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-npp") == 0 then
      i = i + 1
      conf.nodes_per_piece = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-wpp") == 0 then
      i = i + 1
      conf.wires_per_piece = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-pct") == 0 then
      i = i + 1
      conf.pct_wire_in_piece = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-s") == 0 then
      i = i + 1
      conf.random_seed = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-sync") == 0 then
      i = i + 1
      conf.sync = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-prune") == 0 then
      i = i + 1
      conf.prune = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-checks") == 0 then
      conf.perform_checks = true
    elseif cstring.strcmp(args.argv[i], "-dump") == 0 then
      conf.dump_values = true
    end
  end
  return conf
end

terra random_element(arr : &c.legion_ptr_t,
                     num_elmts : uint)
  var index = [uint](drand48() * num_elmts)
  return arr[index]
end

end -- not use_python_main

task init_nodes(rn : region(node))
where
  reads writes(rn)
do
  for node in rn do
    node.node_cap = drand48() + 1.0
    node.leakage = 0.1 * drand48()
    node.charge = 0.0
    node.node_voltage = 2 * drand48() - 1.0
  end
end

task init_wires(spiece_id  : int,
                conf       : Config,
                rgr        : region(ghost_range),
                rpn        : region(node),
                rsn        : region(node),
                all_shared : region(node),
                rw         : region(wire(rpn, rsn, all_shared)))
where
  reads writes(rw, rgr)
do
  var npp = conf.nodes_per_piece
  var snpp = conf.shared_nodes_per_piece
  var pnpp = npp - snpp
  var num_pieces : int = conf.num_pieces
  var num_shared_nodes = num_pieces * snpp
  var pps = conf.pieces_per_superpiece

  var num_neighbors : int = conf.num_neighbors
  if num_neighbors == 0 then
    num_neighbors = [int](ceil((num_pieces - 1) / 2.0 * (conf.density / 100.0)))
  end
  if num_neighbors >= conf.num_pieces then
    num_neighbors = conf.num_pieces - 1
  end

  var window = conf.window
  if window * 2 < num_neighbors then
    window = (num_neighbors + 1) / 2
  end

  var piece_shared_nodes = [&uint](c.malloc([sizeof(uint)] * num_pieces))
  var neighbor_ids : &uint = [&uint](c.malloc([sizeof(uint)] * num_neighbors))
  var alread_picked : &bool = [&bool](c.malloc([sizeof(bool)] * num_pieces))

  var max_shared_node_id = 0
  var min_shared_node_id = num_shared_nodes * pps

  var offset = spiece_id * pps * conf.wires_per_piece

  for piece_id = spiece_id * pps, (spiece_id + 1) * pps do
    var pn_ptr_offset = piece_id * pnpp + num_shared_nodes
    var sn_ptr_offset = piece_id * snpp

    var start_piece_id : int = piece_id - window
    var end_piece_id : int = piece_id + window

    if start_piece_id < 0 then
      start_piece_id = 0
      end_piece_id = min(num_neighbors, num_pieces - 1)
    end
    if end_piece_id >= num_pieces then
      start_piece_id = max(0, (num_pieces - 1) - num_neighbors)
      end_piece_id = num_pieces - 1
    end

    for i = 0, num_pieces do
      piece_shared_nodes[i] = 0
      alread_picked[i] = false
    end

    var window_size = end_piece_id - start_piece_id + 1
    regentlib.assert(start_piece_id >= 0, "wrong start piece id")
    regentlib.assert(end_piece_id < num_pieces, "wrong end piece id")
    regentlib.assert(start_piece_id <= end_piece_id, "wrong neighbor range")
    for i = 0, num_neighbors do
      var neighbor_id = [uint](drand48() * window_size + start_piece_id)
      while neighbor_id == piece_id or alread_picked[neighbor_id] do
        neighbor_id = [uint](drand48() * window_size + start_piece_id)
      end
      alread_picked[neighbor_id] = true
      neighbor_ids[i] = neighbor_id
    end

    for wire_id = 0, conf.wires_per_piece do
      var wire =
        unsafe_cast(ptr(wire(rpn, rsn, all_shared), rw), ptr(wire_id + offset))
      wire.current._0 = 0.0
      wire.current._1 = 0.0
      wire.current._2 = 0.0
      wire.current._3 = 0.0
      wire.current._4 = 0.0
      wire.current._5 = 0.0
      wire.current._6 = 0.0
      wire.current._7 = 0.0
      wire.current._8 = 0.0
      wire.current._9 = 0.0

      wire.voltage._0 = 0.0
      wire.voltage._1 = 0.0
      wire.voltage._2 = 0.0
      wire.voltage._3 = 0.0
      wire.voltage._4 = 0.0
      wire.voltage._5 = 0.0
      wire.voltage._6 = 0.0
      wire.voltage._7 = 0.0
      wire.voltage._8 = 0.0
      wire.resistance = drand48() * 10.0 + 1.0

      -- Keep inductance on the order of 1e-3 * dt to avoid resonance problems
      wire.inductance = (drand48() + 0.1) * DELTAT * 1e-3
      wire.wire_cap = drand48() * 0.1

      var in_node = [uint](drand48() * npp)
      if in_node < snpp then
        in_node += sn_ptr_offset
      else
        in_node += pn_ptr_offset - snpp
      end

      wire.in_ptr = dynamic_cast(ptr(node, rpn, rsn), ptr(in_node))
      regentlib.assert(not isnull(wire.in_ptr), "picked an invalid random pointer")

      var out_node = 0
      if (100 * drand48() < conf.pct_wire_in_piece) or (conf.num_pieces == 1) then
        out_node = [uint](drand48() * npp)
        if out_node < snpp then
          out_node += sn_ptr_offset
        else
          out_node += pn_ptr_offset - snpp
        end
      else
        ---- pick a random other piece and a node from there
        --var pp = [uint](drand48() * (conf.num_pieces - 1))
        --if pp >= piece_id then pp += 1 end

        var pp = neighbor_ids[ [uint](drand48() * num_neighbors) ]
        var idx = [uint](drand48() * snpp)
        if idx >= piece_shared_nodes[pp] then
          idx = piece_shared_nodes[pp]
          if piece_shared_nodes[pp] < snpp then
            piece_shared_nodes[pp] = piece_shared_nodes[pp] + 1
          end
        end
        out_node = pp * snpp + idx
        max_shared_node_id = max(max_shared_node_id, out_node)
        min_shared_node_id = min(min_shared_node_id, out_node)
      end
      wire.out_ptr = dynamic_cast(ptr(node, rpn, rsn, all_shared), ptr(out_node))
    end
    offset += conf.wires_per_piece
  end

  for range in rgr do
    range.rect.lo = min_shared_node_id
    range.rect.hi = max_shared_node_id
  end

  c.free(piece_shared_nodes)
  c.free(neighbor_ids)
  c.free(alread_picked)
end

task init_piece(-- spiece_id   : int,
                conf        : Config,
                rgr         : region(ghost_range),
                rpn         : region(node),
                rsn         : region(node),
                all_shared  : region(node),
                rw          : region(wire(rpn, rsn, all_shared)))
where
  reads writes(rgr, rpn, rsn, rw)
do
  var spiece_id = regentlib.c.legion_logical_region_get_color(__runtime(), __raw(rpn))
  init_nodes(rpn)
  init_nodes(rsn)
  init_wires(spiece_id, conf, rgr, rpn, rsn, all_shared, rw)
end

task init_pointers(rpn : region(node),
                   rsn : region(node),
                   rgn : region(node),
                   rw : region(wire(rpn, rsn, rgn)))
where
  reads writes(rw.{in_ptr, out_ptr})
do
  for w in rw do
    w.in_ptr = dynamic_cast(ptr(node, rpn, rsn), w.in_ptr)
    regentlib.assert(not isnull(w.in_ptr), "in ptr is null!")
    w.out_ptr = dynamic_cast(ptr(node, rpn, rsn, rgn), w.out_ptr)
    regentlib.assert(not isnull(w.out_ptr), "out ptr is null!")
  end
end

__demand(__cuda)
task calculate_new_currents(print_ts : bool,
                            steps : uint,
                            rpn : region(node),
                            rsn : region(node),
                            rgn : region(node),
                            rw : region(wire(rpn, rsn, rgn)),
                            rt : region(timestamp))
where
  reads(rpn.node_voltage, rsn.node_voltage, rgn.node_voltage,
        rw.{in_ptr, out_ptr, inductance, resistance, wire_cap}),
  reads writes(rw.{current, voltage}, rt)
do
  if print_ts then
    var t = c.legion_get_current_time_in_micros()
    for x in rt do x.start = t end
  end

  var dt : float = DELTAT
  var recip_dt : float = 1.0 / dt
  --__demand(__vectorize)
  for w in rw do
    var temp_v : float[WIRE_SEGMENTS + 1]
    var temp_i : float[WIRE_SEGMENTS]
    var old_i : float[WIRE_SEGMENTS]
    var old_v : float[WIRE_SEGMENTS - 1]

    temp_i[0] = w.current._0
    temp_i[1] = w.current._1
    temp_i[2] = w.current._2
    temp_i[3] = w.current._3
    temp_i[4] = w.current._4
    temp_i[5] = w.current._5
    temp_i[6] = w.current._6
    temp_i[7] = w.current._7
    temp_i[8] = w.current._8
    temp_i[9] = w.current._9
    for i = 0, WIRE_SEGMENTS do
      old_i[i] = temp_i[i]
    end

    temp_v[1] = w.voltage._0
    temp_v[2] = w.voltage._1
    temp_v[3] = w.voltage._2
    temp_v[4] = w.voltage._3
    temp_v[5] = w.voltage._4
    temp_v[6] = w.voltage._5
    temp_v[7] = w.voltage._6
    temp_v[8] = w.voltage._7
    temp_v[9] = w.voltage._8
    for i = 0, WIRE_SEGMENTS - 1 do
      old_v[i] = temp_v[i + 1]
    end

    -- Pin the outer voltages to the node voltages
    temp_v[0] = w.in_ptr.node_voltage
    temp_v[WIRE_SEGMENTS] = w.out_ptr.node_voltage

    -- Solve the RLC model iteratively
    var inductance : float = w.inductance
    var recip_resistance : float = 1.0 / w.resistance
    var recip_capacitance : float = 1.0 / w.wire_cap
    for j = 0, steps do
      -- first, figure out the new current from the voltage differential
      -- and our inductance:
      -- dV = R*I + L*I' ==> I = (dV - L*I')/R
      for i = 0, WIRE_SEGMENTS do
        temp_i[i] = ((temp_v[i + 1] - temp_v[i]) -
                     (inductance * (temp_i[i] - old_i[i]) * recip_dt)) * recip_resistance
      end
      -- Now update the inter-node voltages
      for i = 0, WIRE_SEGMENTS - 1 do
        temp_v[i + 1] = old_v[i] + dt * (temp_i[i] - temp_i[i + 1]) * recip_capacitance
      end
    end

    -- Write out the results
    w.current._0 = temp_i[0]
    w.current._1 = temp_i[1]
    w.current._2 = temp_i[2]
    w.current._3 = temp_i[3]
    w.current._4 = temp_i[4]
    w.current._5 = temp_i[5]
    w.current._6 = temp_i[6]
    w.current._7 = temp_i[7]
    w.current._8 = temp_i[8]
    w.current._9 = temp_i[9]

    w.voltage._0 = temp_v[1]
    w.voltage._1 = temp_v[2]
    w.voltage._2 = temp_v[3]
    w.voltage._3 = temp_v[4]
    w.voltage._4 = temp_v[5]
    w.voltage._5 = temp_v[6]
    w.voltage._6 = temp_v[7]
    w.voltage._7 = temp_v[8]
    w.voltage._8 = temp_v[9]
  end
end

__demand(__cuda)
task distribute_charge(rpn : region(node),
                       rsn : region(node),
                       rgn : region(node),
                       rw : region(wire(rpn, rsn, rgn)))
where
  reads(rw.{in_ptr, out_ptr, current._0, current._9}),
  reads writes(rpn.charge),
  reduces +(rsn.charge, rgn.charge)
do
  var dt = DELTAT
  for w in rw do
    var in_current : float = -dt * w.current._0
    var out_current : float = dt * w.current._9
    w.in_ptr.charge += in_current
    w.out_ptr.charge += out_current
  end
end

__demand(__cuda)
task update_voltages(init_ts : bool,
                     print_ts : bool,
                     rpn : region(node),
                     rsn : region(node),
                     rt : region(timestamp))
where
  reads(rpn.{node_cap, leakage}, rsn.{node_cap, leakage}),
  reads writes(rpn.{node_voltage, charge}, rsn.{node_voltage, charge}, rt)
do
  for node in rpn do
    var voltage : float = node.node_voltage + node.charge / node.node_cap
    voltage = voltage * (1.0 - node.leakage)
    node.node_voltage = voltage
    node.charge = 0.0
  end
  for node in rsn do
    var voltage : float = node.node_voltage + node.charge / node.node_cap
    voltage = voltage * (1.0 - node.leakage)
    node.node_voltage = voltage
    node.charge = 0.0
  end

  if init_ts or print_ts then
    var t = c.legion_get_current_time_in_micros()
    if init_ts then
      for x in rt do x.init_stop = t end
    end
    if print_ts then
      for x in rt do x.stop = t end
    end
  end
end

if not use_python_main then

task dump_task(rpn : region(node),
               rsn : region(node),
               rgn : region(node),
               rw : region(wire(rpn, rsn, rgn)))
where
  reads(rpn, rsn, rgn, rw)
do
  for w in rw do
    format.print(" {.5e}", w.current._0);
    format.print(" {.5e}", w.current._1);
    format.print(" {.5e}", w.current._2);
    format.print(" {.5e}", w.current._3);
    format.print(" {.5e}", w.current._4);
    format.print(" {.5e}", w.current._5);
    format.print(" {.5e}", w.current._6);
    format.print(" {.5e}", w.current._7);
    format.print(" {.5e}", w.current._8);
    format.print(" {.5e}", w.current._9);

    format.print(" {.5e}", w.voltage._0);
    format.print(" {.5e}", w.voltage._1);
    format.print(" {.5e}", w.voltage._2);
    format.print(" {.5e}", w.voltage._3);
    format.print(" {.5e}", w.voltage._4);
    format.print(" {.5e}", w.voltage._5);
    format.print(" {.5e}", w.voltage._6);
    format.print(" {.5e}", w.voltage._7);
    format.print(" {.5e}", w.voltage._8);

    format.println("");
  end
end

__demand(__inline)
task create_colorings(conf : Config)
  var coloring : Colorings
  coloring.privacy_map = c.legion_point_coloring_create()
  coloring.private_node_map = c.legion_point_coloring_create()
  coloring.shared_node_map = c.legion_point_coloring_create()
  var num_circuit_nodes : uint64 = conf.num_pieces * conf.nodes_per_piece
  var num_shared_nodes = conf.num_pieces * conf.shared_nodes_per_piece

  regentlib.assert(
    (num_circuit_nodes - num_shared_nodes) % conf.num_pieces == 0,
    "something went wrong in the arithmetic")

  c.legion_point_coloring_add_range(coloring.privacy_map, ptr(1),
    c.legion_ptr_t { value = 0 },
    c.legion_ptr_t { value = num_shared_nodes - 1})

  c.legion_point_coloring_add_range(coloring.privacy_map, ptr(0),
    c.legion_ptr_t { value = num_shared_nodes },
    c.legion_ptr_t { value = num_circuit_nodes - 1})

  var pps = conf.pieces_per_superpiece
  var num_superpieces = conf.num_pieces / pps
  var snpp = conf.shared_nodes_per_piece
  var pnpp = conf.nodes_per_piece - snpp
  for spiece_id = 0, num_superpieces do
    c.legion_point_coloring_add_range(coloring.shared_node_map, ptr(spiece_id),
      c.legion_ptr_t { value = spiece_id * snpp * pps },
      c.legion_ptr_t { value = (spiece_id + 1) * snpp * pps - 1})
    c.legion_point_coloring_add_range(coloring.private_node_map, ptr(spiece_id),
      c.legion_ptr_t { value = num_shared_nodes + spiece_id * pnpp * pps},
      c.legion_ptr_t { value = num_shared_nodes + (spiece_id + 1) * pnpp * pps - 1})
  end
  return coloring
end

task create_ghost_partition(conf         : Config,
                            all_shared   : region(node),
                            ghost_ranges : region(ghost_range))
where
  reads(ghost_ranges)
do
  var ghost_node_map = c.legion_point_coloring_create()
  var num_superpieces = conf.num_pieces / conf.pieces_per_superpiece

  for range in ghost_ranges do
    c.legion_point_coloring_add_range(ghost_node_map,
      range,
      c.legion_ptr_t { value = range.rect.lo },
      c.legion_ptr_t { value = range.rect.hi })
  end

  return partition(aliased, all_shared, ghost_node_map, ghost_ranges.ispace)
end

task parse_input(conf : Config)
  conf = parse_input_args(conf)

  regentlib.assert(conf.num_pieces % conf.pieces_per_superpiece == 0,
      "pieces should be evenly distributed to superpieces")
  conf.shared_nodes_per_piece =
    [int](ceil(conf.nodes_per_piece * conf.pct_shared_nodes / 100.0))
  format.println("circuit settings: loops={} prune={} pieces={} (pieces/superpiece={}) nodes/piece={} (nodes/piece={}) wires/piece={} pct_in_piece={} seed={}",
    conf.num_loops, conf.prune, conf.num_pieces, conf.pieces_per_superpiece, conf.nodes_per_piece,
    conf.shared_nodes_per_piece, conf.wires_per_piece, conf.pct_wire_in_piece, conf.random_seed)

  var num_pieces = conf.num_pieces
  var num_circuit_nodes : uint64 = num_pieces * conf.nodes_per_piece
  var num_circuit_wires : uint64 = num_pieces * conf.wires_per_piece

  -- report mesh size in bytes
  do
    var node_size = [ terralib.sizeof(node) ]
    var wire_size = [ terralib.sizeof(wire(wild,wild,wild)) ]
    format.println("Circuit memory usage:")
    format.println("  Nodes : {10} * {4} bytes = {12} bytes", num_circuit_nodes, node_size, num_circuit_nodes * node_size)
    format.println("  Wires : {10} * {4} bytes = {12} bytes", num_circuit_wires, wire_size, num_circuit_wires * wire_size)
    var total = ((num_circuit_nodes * node_size) + (num_circuit_wires * wire_size))
    format.println("  Total                             {12} bytes", total)
  end

  return conf
end

task begin_init(rt : region(timestamp))
where writes(rt) do
  var t = c.legion_get_current_time_in_micros()
  for x in rt do x.init_start = t end
end

task get_elapsed(all_times : region(timestamp))
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

task print_summary(color : int, init_time : double, sim_time : double, conf : Config)
  if color == 0 then
    format.println("INIT TIME = {7.3} s", init_time)
    format.println("ELAPSED TIME = {7.3} s", sim_time)

    -- Compute the floating point operations per second
    var num_circuit_nodes : uint64 = conf.num_pieces * conf.nodes_per_piece
    var num_circuit_wires : uint64 = conf.num_pieces * conf.wires_per_piece
    -- calculate currents
    var operations : uint64 = num_circuit_wires * (WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * conf.steps
    -- distribute charge
    operations = operations + (num_circuit_wires * 4)
    -- update voltages
    operations = operations + (num_circuit_nodes * 4)
    -- multiply by the number of loops
    operations = operations * conf.num_loops

    -- Compute the number of gflops
    var gflops = (1e-9*operations)/sim_time
    format.println("GFLOPS = {7.3} GFLOPS", gflops)
  end
end

__demand(__inner, __replicable)
task toplevel()
  var conf : Config
  conf.num_loops = 5
  conf.num_pieces = 4
  conf.pieces_per_superpiece = 1
  conf.nodes_per_piece = 4
  conf.wires_per_piece = 8
  conf.pct_wire_in_piece = 80
  conf.random_seed = 12345
  conf.steps = STEPS
  conf.sync = 0
  conf.prune = 0
  conf.perform_checks = false
  conf.dump_values = false
  conf.pct_shared_nodes = 1.0
  conf.density = 20 -- ignored if num_neighbors > 0
  conf.num_neighbors = 5 -- set 0 if density parameter is to be picked
  conf.window = 3 -- find neighbors among [piece_id - window, piece_id + window]

  conf = parse_input(conf)

  var num_pieces = conf.num_pieces
  var num_superpieces = conf.num_pieces / conf.pieces_per_superpiece
  var num_circuit_nodes : uint64 = num_pieces * conf.nodes_per_piece
  var num_circuit_wires : uint64 = num_pieces * conf.wires_per_piece

  var launch_domain = ispace(ptr, num_superpieces)
  var all_times = region(ispace(ptr, num_superpieces), timestamp)
  fill(all_times.{init_start, init_stop, start, stop}, 0)
  var rp_times = partition(equal, all_times, launch_domain)

  __fence(__execution, __block)
  __demand(__index_launch)
  for i in launch_domain do
    begin_init(rp_times[i])
  end
  __fence(__execution, __block)

  var all_nodes = region(ispace(ptr, num_circuit_nodes), node)
  var all_wires = region(ispace(ptr, num_circuit_wires), wire(wild, wild, wild))

  fill(all_nodes.{node_cap, leakage, charge, node_voltage}, 0.0)
  fill(all_wires.{inductance, resistance, wire_cap, current.{_0, _1, _2, _3, _4, _5, _6, _7, _8, _9}, voltage.{_0, _1, _2, _3, _4, _5, _6, _7, _8}}, 0.0)

  var colorings = create_colorings(conf)
  var rp_all_nodes = partition(disjoint, all_nodes, colorings.privacy_map, ispace(ptr, 2))
  var all_private = rp_all_nodes[0]
  var all_shared = rp_all_nodes[1]

  var rp_private = partition(disjoint, all_private, colorings.private_node_map, launch_domain)
  var rp_shared = partition(disjoint, all_shared, colorings.shared_node_map, launch_domain)
  var rp_wires = partition(equal, all_wires, launch_domain)

  var ghost_ranges = region(ispace(ptr, num_superpieces), ghost_range)
  var rp_ghost_ranges = partition(equal, ghost_ranges, launch_domain)

  fill(ghost_ranges.rect, rect1d { 0, 0 })

  for j = 0, 1 do
    __demand(__index_launch)
    for i in launch_domain do
      init_piece(conf, rp_ghost_ranges[i],
                 rp_private[i], rp_shared[i], all_shared, rp_wires[i])
    end
  end

  var rp_ghost = create_ghost_partition(conf, all_shared, ghost_ranges)

  __demand(__spmd)
  for j = 0, 1 do
    for i in launch_domain do
      init_pointers(rp_private[i], rp_shared[i], rp_ghost[i], rp_wires[i])
    end
  end

  var simulation_success = true
  var steps = conf.steps
  var prune = conf.prune
  var num_loops = conf.num_loops + 2*prune

  __fence(__execution, __block)
  __demand(__spmd, __trace)
  for j = 0, num_loops do
    for i in launch_domain do
      calculate_new_currents(j == prune, steps, rp_private[i], rp_shared[i], rp_ghost[i], rp_wires[i], rp_times[i])
    end
    for i in launch_domain do
      distribute_charge(rp_private[i], rp_shared[i], rp_ghost[i], rp_wires[i])
    end
    for i in launch_domain do
      update_voltages(j == 0, j == num_loops - prune - 1, rp_private[i], rp_shared[i], rp_times[i])
    end
  end

  var { init_time, sim_time } = get_elapsed(all_times)
  for i = 0, num_superpieces do print_summary(i, init_time, sim_time, conf) end
end

else -- not use_python_main

extern task toplevel()
toplevel:set_task_id(2)

end -- not use_python_main

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
  local link_flags = terralib.newlist({"-L" .. out_dir, "-lcircuit_mapper", "-lm"})

  if os.getenv('STANDALONE') == '1' then
    os.execute('cp ' .. os.getenv('LG_RT_DIR') .. '/../bindings/regent/' ..
        regentlib.binding_library .. ' ' .. out_dir)
  end

  local exe = os.getenv('OBJNAME') or "circuit"
  regentlib.saveobj(toplevel, exe, "executable", cmapper.register_mappers, link_flags)
else
  regentlib.start(toplevel, cmapper.register_mappers)
end
