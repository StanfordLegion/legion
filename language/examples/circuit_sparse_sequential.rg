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

-- runs-with:
-- [
--   ["-ll:cpu", "5", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "5", "-p", "5"]
-- ]

import "regent"

-- Compile and link circuit.cc
local ccircuit
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
  local circuit_cc = root_dir .. "circuit.cc"
  local circuit_so
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    circuit_so = out_dir .. "libcircuit.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    circuit_so = root_dir .. "libcircuit.so"
  else
    circuit_so = os.tmpname() .. ".so" -- root_dir .. "circuit.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CC_FLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 circuit_cc .. " -o " .. circuit_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. circuit_cc)
    assert(false)
  end
  terralib.linklibrary(circuit_so)
  ccircuit = terralib.includec("circuit.h", {"-I", root_dir, "-I", runtime_dir})
end

local c = regentlib.c
local cmath = terralib.includec("math.h")
local cstring = terralib.includec("string.h")
local drand48 = regentlib.c.drand48
local srand48 = regentlib.c.srand48
local atoi = regentlib.c.atoi
local ceil = regentlib.ceil(double)

WIRE_SEGMENTS = 10
STEPS = 10000
DELTAT = 1e-6

struct Colorings {
  privacy_map : c.legion_point_coloring_t,
  private_node_map : c.legion_point_coloring_t,
  shared_node_map : c.legion_point_coloring_t,
}

struct Config {
  num_loops : uint,
  num_pieces : uint,
  pieces_per_superpiece : uint,
  nodes_per_piece : uint,
  wires_per_piece : uint,
  pct_wire_in_piece : uint,
  random_seed : uint,
  steps : uint,
  sync : uint,
  prune : uint,
  perform_checks : bool,
  dump_values : bool,
  pct_shared_nodes : double,
  shared_nodes_per_piece : uint,
  density : uint,
  num_neighbors : uint,
  window : int,
}

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
  first : int,
  last : int,
}

fspace wire {
  inN : ptr,
  outN : ptr,
  inductance : float,
  resistance : float,
  wire_cap : float,
  current : currents,
  voltage : voltages,
}

terra parse_input_args(conf : Config)
  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstring.strcmp(args.argv[i], "-l") == 0 then
      i = i + 1
      conf.num_loops = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-i") == 0 then
      i = i + 1
      conf.steps = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-p") == 0 then
      i = i + 1
      conf.num_pieces = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-pps") == 0 then
      i = i + 1
      conf.pieces_per_superpiece = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-npp") == 0 then
      i = i + 1
      conf.nodes_per_piece = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-wpp") == 0 then
      i = i + 1
      conf.wires_per_piece = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-pct") == 0 then
      i = i + 1
      conf.pct_wire_in_piece = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-s") == 0 then
      i = i + 1
      conf.random_seed = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-sync") == 0 then
      i = i + 1
      conf.sync = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-prune") == 0 then
      i = i + 1
      conf.prune = atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-checks") == 0 then
      conf.perform_checks = true
    elseif cstring.strcmp(args.argv[i], "-dump") == 0 then
      conf.dump_values = true
    end
  end
  return conf
end

__demand(__parallel)
task init_nodes(rn : region(ispace(ptr), node))
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
                rw         : region(ispace(ptr), wire))
where
  reads writes(rw)
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
      var wire = unsafe_cast(ptr(wire, rw), [ptr](wire_id + offset))
      wire.current.{_0, _1, _2, _3, _4, _5, _6, _7, _8, _9} = 0.0
      wire.voltage.{_0, _1, _2, _3, _4, _5, _6, _7, _8} = 0.0
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

      wire.inN = [ptr](in_node)

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
      wire.outN = [ptr](out_node)
    end
    offset += conf.wires_per_piece
  end

  c.free(piece_shared_nodes)
  c.free(neighbor_ids)
  c.free(alread_picked)
end

task init_piece(spiece_id : int,
                conf      : Config,
                rpn       : region(ispace(ptr), node),
                rsn       : region(ispace(ptr), node),
                rw        : region(ispace(ptr), wire))
where
  reads writes(rpn, rsn, rw)
do
  init_nodes(rpn)
  init_nodes(rsn)
  init_wires(spiece_id, conf, rw)
end

__demand(__cuda, __parallel)
task calculate_new_currents(steps : uint,
                            rn : region(ispace(ptr), node),
                            rw : region(ispace(ptr), wire))
where
  reads(rn.node_voltage,
        rw.{inN, outN, inductance, resistance, wire_cap}),
  reads writes(rw.{current, voltage})
do
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
    var inN = rw[w].inN
    var in_voltage = rn[inN].node_voltage
    temp_v[0] = in_voltage
    var outN = rw[w].outN
    var out_voltage = rn[outN].node_voltage
    temp_v[WIRE_SEGMENTS] = out_voltage

    -- Solve the RLC model iteratively
    var inductance : float = rw[w].inductance
    var resistance : float = rw[w].resistance
    var recip_resistance : float = 1.0 / resistance
    var wire_cap : float = rw[w].wire_cap
    var recip_capacitance : float = 1.0 / wire_cap
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
    rw[w].current._0 = temp_i[0]
    rw[w].current._1 = temp_i[1]
    rw[w].current._2 = temp_i[2]
    rw[w].current._3 = temp_i[3]
    rw[w].current._4 = temp_i[4]
    rw[w].current._5 = temp_i[5]
    rw[w].current._6 = temp_i[6]
    rw[w].current._7 = temp_i[7]
    rw[w].current._8 = temp_i[8]
    rw[w].current._9 = temp_i[9]

    rw[w].voltage._0 = temp_v[1]
    rw[w].voltage._1 = temp_v[2]
    rw[w].voltage._2 = temp_v[3]
    rw[w].voltage._3 = temp_v[4]
    rw[w].voltage._4 = temp_v[5]
    rw[w].voltage._5 = temp_v[6]
    rw[w].voltage._6 = temp_v[7]
    rw[w].voltage._7 = temp_v[8]
    rw[w].voltage._8 = temp_v[9]
  end
end

__demand(__cuda, __parallel)
task distribute_charge(rn : region(ispace(ptr), node),
                       rw : region(ispace(ptr), wire))
where
  reads(rw.{inN, outN, current._0, current._9}),
  reduces +(rn.charge)
do
  var dt = DELTAT
  for w in rw do
    var in_current : float = rw[w].current._0
    var out_current : float = rw[w].current._9
    var in_current_dt = -dt * in_current
    var out_current_dt = dt * out_current
    var inN = rw[w].inN
    var outN = rw[w].outN
    rn[inN].charge += in_current_dt
    rn[outN].charge += out_current_dt
  end
end

__demand(__cuda, __parallel)
task update_voltages(rn : region(ispace(ptr), node))
where
  reads(rn.{node_cap, leakage}),
  reads writes(rn.{node_voltage, charge})
do
  for node in rn do
    var node_voltage = rn[node].node_voltage
    var charge = rn[node].charge
    var node_cap = rn[node].node_cap
    var leakage = rn[node].leakage
    var voltage : float = node_voltage + charge / node_cap
    voltage = voltage * (1.0 - leakage)
    rn[node].node_voltage = voltage
    rn[node].charge = 0.0
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

task parse_input(conf : Config)
  return parse_input_args(conf)
end

task print_summary(color : int, sim_time : double, conf : Config)
  if color == 0 then
    c.printf("ELAPSED TIME = %7.3f s\n", sim_time)

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
    c.printf("GFLOPS = %7.3f GFLOPS\n", gflops)
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
  regentlib.assert(conf.num_pieces % conf.pieces_per_superpiece == 0,
      "pieces should be evenly distributed to superpieces")
  conf.shared_nodes_per_piece =
    [int](ceil(conf.nodes_per_piece * conf.pct_shared_nodes / 100.0))
  --c.printf("circuit settings: loops=%d prune=%d pieces=%d (pieces/superpiece=%d) nodes/piece=%d (nodes/piece=%d) wires/piece=%d pct_in_piece=%d seed=%d\n",
  --  conf.num_loops, conf.prune, conf.num_pieces, conf.pieces_per_superpiece, conf.nodes_per_piece,
  --  conf.shared_nodes_per_piece, conf.wires_per_piece, conf.pct_wire_in_piece, conf.random_seed)

  var num_pieces = conf.num_pieces
  var num_superpieces = conf.num_pieces / conf.pieces_per_superpiece
  var num_circuit_nodes : uint64 = num_pieces * conf.nodes_per_piece
  var num_circuit_wires : uint64 = num_pieces * conf.wires_per_piece

  var rn = region(ispace(ptr, num_circuit_nodes), node)
  var rw = region(ispace(ptr, num_circuit_wires), wire)

  -- report mesh size in bytes
  --do
  --  var node_size = [ terralib.sizeof(node) ]
  --  var wire_size = [ terralib.sizeof(wire) ]
  --  c.printf("Circuit memory usage:\n")
  --  c.printf("  Nodes : %10lld * %4d bytes = %12lld bytes\n", num_circuit_nodes, node_size, num_circuit_nodes * node_size)
  --  c.printf("  Wires : %10lld * %4d bytes = %12lld bytes\n", num_circuit_wires, wire_size, num_circuit_wires * wire_size)
  --  var total = ((num_circuit_nodes * node_size) + (num_circuit_wires * wire_size))
  --  c.printf("  Total                             %12lld bytes\n", total)
  --end

  var color_space = ispace(ptr, num_superpieces)
  var p_rw = partition(equal, rw, color_space)
  var colorings = create_colorings(conf)
  var rp_all_nodes = partition(disjoint, rn, colorings.privacy_map, ispace(ptr, 2))
  var all_private = rp_all_nodes[0]
  var all_shared = rp_all_nodes[1]

  var pn_private = partition(disjoint, all_private, colorings.private_node_map, color_space)
  var pn_shared = partition(disjoint, all_shared, colorings.shared_node_map, color_space)

  for color in color_space do
    init_piece([int](color), conf, pn_private[color], pn_shared[color], p_rw[color])
  end

  var simulation_success = true
  var steps = conf.steps
  var prune = conf.prune
  var num_loops = conf.num_loops + 2*prune

  var ts_start = c.legion_get_current_time_in_micros()
  var ts_end = ts_start

  __parallelize_with color_space, p_rw,
                     complete(pn_private | pn_shared, rn),
                     disjoint(pn_private | pn_shared)
  do
    __demand(__spmd)
    for j = 0, num_loops do
      if j == prune then
        __fence(__execution, __block)
        ts_start = c.legion_get_current_time_in_micros()
      end
      calculate_new_currents(steps, rn, rw)
      distribute_charge(rn, rw)
      update_voltages(rn)
      if j == num_loops - prune - 1 then
        __fence(__execution, __block)
        ts_end = c.legion_get_current_time_in_micros()
      end
    end
  end

  var sim_time = 1e-6 * (ts_end - ts_start)
  for i = 0, num_superpieces do print_summary(i, sim_time, conf) end
end

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
  local link_flags = terralib.newlist({"-L" .. out_dir, "-lcircuit", "-lm"})

  if os.getenv('STANDALONE') == '1' then
    os.execute('cp ' .. os.getenv('LG_RT_DIR') .. '/../bindings/regent/libregent.so ' .. out_dir)
  end

  local exe = os.getenv('OBJNAME') or "circuit"
  regentlib.saveobj(toplevel, exe, "executable", ccircuit.register_mappers, link_flags)
else
  regentlib.start(toplevel, ccircuit.register_mappers)
end
