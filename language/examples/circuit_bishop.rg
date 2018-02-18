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
--   ["-ll:cpu", "4", "-fcuda", "0"]
-- ]
import "regent"
import "bishop"

mapper

$CPUs = processors[isa=x86]
$GPUs = processors[isa=cuda]
$HAS_GPUS = $GPUs.size > 0

task#calculate_new_currents[index=$i],
task#distribute_charge[index=$i],
task#update_voltages[index=$i] {
  target : $HAS_GPUS ? $GPUs[$i % $GPUs.size] : $CPUs[$i % $CPUs.size];
}

task#block_task[index=$i] {
  target : $HAS_GPUS ? $GPUs[$i % $GPUs.size] : $CPUs[$i % $CPUs.size];
}
task[isa=cuda and target=$p] region {
  target : $p.memories[kind=fbmem];
}

task[isa=x86 and target=$p] region {
  target : $p.memories[kind=sysmem];
}

task#calculate_new_currents[isa=cuda and target=$p] region#rsn {
  target : $p.memories[kind=zcmem];
}

task#calculate_new_currents[isa=cuda and target=$p] region#rgn {
  target : $p.memories[kind=zcmem];
}

task#distribute_charge[isa=cuda and target=$p] region#rsn {
  target : $p.memories[kind=zcmem];
}

task#distribute_charge[isa=cuda and target=$p] region#rgn {
  target : $p.memories[kind=zcmem];
}

task#update_voltages[isa=cuda and target=$p] region#rsn {
  target : $p.memories[kind=zcmem];
}

task#block_task[isa=cuda and target=$p] region#rsn {
  target : $p.memories[kind=zcmem];
}

task#block_task[isa=cuda and target=$p] region#rgn {
  target : $p.memories[kind=zcmem];
}

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
  privacy_map : c.legion_coloring_t,
  private_node_map : c.legion_coloring_t,
  shared_node_map : c.legion_coloring_t,
}

struct Config {
  num_loops : uint,
  num_pieces : uint,
  nodes_per_piece : uint,
  wires_per_piece : uint,
  pct_wire_in_piece : uint,
  random_seed : uint,
  steps : uint,
  sync : uint,
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

task init_wires(piece_id   : int,
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
  var num_pieces = conf.num_pieces
  var num_shared_nodes = num_pieces * snpp

  var pn_ptr_offset = piece_id * pnpp + num_shared_nodes
  var sn_ptr_offset = piece_id * snpp

  var num_neighbors : int = conf.num_neighbors
  if num_neighbors == 0 then
    num_neighbors = [int](ceil((num_pieces - 1) / 2.0 * (conf.density / 100.0)))
  end
  if num_neighbors >= conf.num_pieces then
    num_neighbors = conf.num_pieces - 1
  end
  var neighbor_ids : &uint = [&uint](c.malloc([sizeof(uint)] * num_neighbors))
  var alread_picked : &bool = [&bool](c.malloc([sizeof(bool)] * num_pieces))

  var piece_shared_nodes = [&uint](c.malloc([sizeof(uint)] * conf.num_pieces))

  var window = conf.window
  if window * 2 < num_neighbors then
    window = (num_neighbors + 1) / 2
  end

  var start_piece_id = piece_id - window
  var end_piece_id = piece_id + window

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
  for i = 0, num_neighbors do
    var neighbor_id = [uint](drand48() * window_size + start_piece_id)
    while neighbor_id == piece_id or alread_picked[neighbor_id] do
      neighbor_id = [uint](drand48() * window_size + start_piece_id)
    end
    alread_picked[neighbor_id] = true
    neighbor_ids[i] = neighbor_id
  end

  var max_shared_node_id = 0
  var min_shared_node_id = num_shared_nodes
  for wire in rw do
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

    wire.in_ptr = dynamic_cast(ptr(node, rpn, rsn), [ptr](in_node))
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
    wire.out_ptr = dynamic_cast(ptr(node, rpn, rsn, all_shared), [ptr](out_node))
  end

  for range in rgr do
    range.first = min_shared_node_id
    range.last = max_shared_node_id
  end
  c.free(piece_shared_nodes)
end

task init_piece(piece_id    : int,
                conf        : Config,
                rgr         : region(ghost_range),
                rpn         : region(node),
                rsn         : region(node),
                all_shared  : region(node),
                rw          : region(wire(rpn, rsn, all_shared)))
where
  reads writes(rgr, rpn, rsn, rw)
do
  init_nodes(rpn)
  init_nodes(rsn)
  init_wires(piece_id, conf, rgr, rpn, rsn, all_shared, rw)
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
task calculate_new_currents(steps : uint,
                            rpn : region(node),
                            rsn : region(node),
                            rgn : region(node),
                            rw : region(wire(rpn, rsn, rgn)))
where
  reads(rpn.node_voltage, rsn.node_voltage, rgn.node_voltage,
        rw.{in_ptr, out_ptr, inductance, resistance, wire_cap}),
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
task update_voltages(rpn : region(node),
                     rsn : region(node))
where
  reads(rpn.{node_cap, leakage}, rsn.{node_cap, leakage}),
  reads writes(rpn.{node_voltage, charge}, rsn.{node_voltage, charge})
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
end

__demand(__cuda)
task block_task(rpn : region(node),
                rsn : region(node),
                rgn : region(node),
                rw : region(wire(rpn, rsn, rgn))) : int64
where reads(rpn, rsn, rgn), reads writes(rw) do
  return [int64](c.legion_get_current_time_in_micros())
end

terra wait_for(x : int)
  return x
end

terra create_colorings(conf : Config)
  var coloring : Colorings
  coloring.privacy_map = c.legion_coloring_create()
  coloring.private_node_map = c.legion_coloring_create()
  coloring.shared_node_map = c.legion_coloring_create()
  var num_circuit_nodes : uint64 = conf.num_pieces * conf.nodes_per_piece
  var num_shared_nodes = conf.num_pieces * conf.shared_nodes_per_piece

  regentlib.assert(
    (num_circuit_nodes - num_shared_nodes) % conf.num_pieces == 0,
    "something went wrong in the arithmetic")

  c.legion_coloring_add_range(coloring.privacy_map, 1,
    c.legion_ptr_t { value = 0 },
    c.legion_ptr_t { value = num_shared_nodes - 1})

  c.legion_coloring_add_range(coloring.privacy_map, 0,
    c.legion_ptr_t { value = num_shared_nodes },
    c.legion_ptr_t { value = num_circuit_nodes - 1})

  var snpp = conf.shared_nodes_per_piece
  var pnpp = conf.nodes_per_piece - snpp
  for piece_id = 0, conf.num_pieces do
    c.legion_coloring_add_range(coloring.shared_node_map, piece_id,
      c.legion_ptr_t { value = piece_id * snpp },
      c.legion_ptr_t { value = (piece_id + 1) * snpp - 1})
    c.legion_coloring_add_range(coloring.private_node_map, piece_id,
      c.legion_ptr_t { value = num_shared_nodes + piece_id * pnpp},
      c.legion_ptr_t { value = num_shared_nodes + (piece_id + 1) * pnpp - 1})
  end
  return coloring
end

task create_ghost_partition(conf         : Config,
                            all_shared   : region(node),
                            ghost_ranges : region(ghost_range))
where
  reads(ghost_ranges)
do
  var ghost_node_map = c.legion_coloring_create()
  var num_pieces = conf.num_pieces
  for i = 0, num_pieces do
    c.legion_coloring_ensure_color(ghost_node_map, i)
  end

  var idx = 0
  for range in ghost_ranges do
    c.legion_coloring_add_range(ghost_node_map,
      idx,
      c.legion_ptr_t { value = range.first },
      c.legion_ptr_t { value = range.last })
    idx += 1
  end

  return partition(aliased, all_shared, ghost_node_map)
end

task toplevel()
  var conf : Config
  conf.num_loops = 5
  conf.num_pieces = 4
  conf.nodes_per_piece = 4
  conf.wires_per_piece = 8
  conf.pct_wire_in_piece = 80
  conf.random_seed = 12345
  conf.steps = STEPS
  conf.sync = 0
  conf.perform_checks = false
  conf.dump_values = false
  conf.pct_shared_nodes = 20.0
  conf.density = 20 -- ignored if num_neighbors > 0
  conf.num_neighbors = 5 -- set 0 if density parameter is to be picked
  conf.window = 3 -- find neighbors among [piece_id - window, piece_id + window]

  conf = parse_input_args(conf)
  conf.shared_nodes_per_piece =
    [int](ceil(conf.nodes_per_piece * conf.pct_shared_nodes / 100.0))
  c.printf("circuit settings: loops=%d pieces=%d nodes/piece=%d (nodes/piece=%d) wires/piece=%d pct_in_piece=%d seed=%d\n",
    conf.num_loops, conf.num_pieces, conf.nodes_per_piece, conf.shared_nodes_per_piece,
    conf.wires_per_piece, conf.pct_wire_in_piece, conf.random_seed)

  var num_pieces = conf.num_pieces
  var num_circuit_nodes : uint64 = num_pieces * conf.nodes_per_piece
  var num_circuit_wires : uint64 = num_pieces * conf.wires_per_piece

  var all_nodes = region(ispace(ptr, num_circuit_nodes), node)
  var all_wires = region(ispace(ptr, num_circuit_wires), wire(wild, wild, wild))

  -- report mesh size in bytes
  do
    var node_size = [ terralib.sizeof(node) ]
    var wire_size = [ terralib.sizeof(wire(wild,wild,wild)) ]
    c.printf("Circuit memory usage:\n")
    c.printf("  Nodes : %10lld * %4d bytes = %12lld bytes\n", num_circuit_nodes, node_size, num_circuit_nodes * node_size)
    c.printf("  Wires : %10lld * %4d bytes = %12lld bytes\n", num_circuit_wires, wire_size, num_circuit_wires * wire_size)
    var total = ((num_circuit_nodes * node_size) + (num_circuit_wires * wire_size))
    c.printf("  Total                             %12lld bytes\n", total)
  end

  var colorings = create_colorings(conf)
  var rp_all_nodes = partition(disjoint, all_nodes, colorings.privacy_map)
  var all_private = rp_all_nodes[0]
  var all_shared = rp_all_nodes[1]

  var launch_domain = ispace(int1d, num_pieces)
  var rp_private = partition(disjoint, all_private, colorings.private_node_map)
  var rp_shared = partition(disjoint, all_shared, colorings.shared_node_map)
  var rp_wires = partition(equal, all_wires, launch_domain)

  var ghost_ranges = region(ispace(ptr, num_pieces), ghost_range)
  var rp_ghost_ranges = partition(equal, ghost_ranges, launch_domain)

  for i = 0, num_pieces do
    init_piece(i, conf, rp_ghost_ranges[i],
               rp_private[i], rp_shared[i], all_shared, rp_wires[i])
  end

  var rp_ghost = create_ghost_partition(conf, all_shared, ghost_ranges)

  for i = 0, num_pieces do
    init_pointers(rp_private[i], rp_shared[i], rp_ghost[i], rp_wires[i])
  end

  c.printf("Starting main simulation loop\n")
  var steps = conf.steps
  var num_loops = conf.num_loops

  var ts_start : int64 = 0x7FFFFFFFFFFFFFFFL
  var ts_stop : int64 = 0

  for i = 0, num_pieces do
    ts_start min= block_task(rp_private[i], rp_shared[i], rp_ghost[i], rp_wires[i])
  end

  __demand(__spmd)
  for j = 0, num_loops do
    regentlib.c.legion_runtime_begin_trace(__runtime(), __context(), 0, false)
    for i = 0, num_pieces do
      calculate_new_currents(steps, rp_private[i], rp_shared[i], rp_ghost[i], rp_wires[i])
    end
    for i = 0, num_pieces do
      distribute_charge(rp_private[i], rp_shared[i], rp_ghost[i], rp_wires[i])
    end
    for i = 0, num_pieces do
      update_voltages(rp_private[i], rp_shared[i])
    end
    regentlib.c.legion_runtime_end_trace(__runtime(), __context(), 0)
  end

  for i = 0, num_pieces do
    ts_stop max= block_task(rp_private[i], rp_shared[i], rp_ghost[i], rp_wires[i])
  end

  do
    var sim_time : double = 1e-6 * (ts_stop - ts_start)
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
  c.printf("simulation complete - destroying regions\n")
end

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local link_flags = terralib.newlist({"-lm"})
  regentlib.saveobj(toplevel, "circuit", "executable", bishoplib.make_entry(), link_flags)
else
  regentlib.start(toplevel, bishoplib.make_entry())
end
