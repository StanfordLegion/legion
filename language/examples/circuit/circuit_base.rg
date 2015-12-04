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

import "regent"

local c = regentlib.c
local std = terralib.includec("stdlib.h")
local cstring = terralib.includec("string.h")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)

WIRE_SEGMENTS = 3
DT = 1e-6

struct Config {
  num_loops : uint,
  num_pieces : uint,
  nodes_per_piece : uint,
  wires_per_piece : uint,
  pct_wire_in_piece : uint,
  random_seed : uint,
  steps : uint,
}

fspace Node {
  node_cap      : float,
  leakage       : float,
  charge        : float,
  node_voltage  : float,
}

struct Currents {
  _0 : float,
  _1 : float,
  _2 : float,
}

struct Voltages {
  _0 : float,
  _1 : float,
}

fspace Wire(rn : region(Node)) {
  in_ptr     : ptr(Node, rn),
  out_ptr    : ptr(Node, rn),
  inductance : float,
  resistance : float,
  wire_cap   : float,
  current    : Currents,
  voltage    : Voltages,
}

terra parse_input_args(conf : Config)
  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstring.strcmp(args.argv[i], "-l") == 0 then
      conf.num_loops = std.atoi(args.argv[i + 1])
    elseif cstring.strcmp(args.argv[i], "-i") == 0 then
      conf.steps = std.atoi(args.argv[i + 1])
    elseif cstring.strcmp(args.argv[i], "-p") == 0 then
      conf.num_pieces = std.atoi(args.argv[i + 1])
    elseif cstring.strcmp(args.argv[i], "-npp") == 0 then
      conf.nodes_per_piece = std.atoi(args.argv[i + 1])
    elseif cstring.strcmp(args.argv[i], "-wpp") == 0 then
      conf.wires_per_piece = std.atoi(args.argv[i + 1])
    elseif cstring.strcmp(args.argv[i], "-pct") == 0 then
      conf.pct_wire_in_piece = std.atoi(args.argv[i + 1])
    elseif cstring.strcmp(args.argv[i], "-s") == 0 then
      conf.random_seed = std.atoi(args.argv[i + 1])
    end
  end
  return conf
end

terra random_element(arr : &c.legion_ptr_t,
                     num_elmts : uint)
  var index = [uint](drand48() * num_elmts)
  return arr[index]
end

task load_circuit(rn : region(Node),
                  rw : region(Wire(rn)),
                  conf : Config)
where reads writes(rn, rw)
do
  var piece_shared_nodes : &uint =
    [&uint](c.malloc([sizeof(uint)] * conf.num_pieces))
  for i = 0, conf.num_pieces do piece_shared_nodes[i] = 0 end

  srand48(conf.random_seed)

  for p = 0, conf.num_pieces do
    for i = 0, conf.nodes_per_piece do
      var node = new(ptr(Node, rn))
      node.node_cap = drand48() + 1.0
      node.leakage = 0.1 * drand48()
      node.charge = 0.0
      node.node_voltage = 2 * drand48() - 1.0
    end
  end

  for p = 0, conf.num_pieces do
    var ptr_offset = p * conf.nodes_per_piece
    for i = 0, conf.wires_per_piece do
      var wire = new(ptr(Wire(rn), rw))
      wire.current.{_0, _1, _2} = 0.0
      wire.voltage.{_0, _1} = 0.0
      wire.resistance = drand48() * 10.0 + 1.0
      -- Keep inductance on the order of 1e-3 * dt to avoid resonance problems
      wire.inductance = (drand48() + 0.1) * DT * 1e-3
      wire.wire_cap = drand48() * 0.1

      var in_node = ptr_offset + [uint](drand48() * conf.nodes_per_piece)
      wire.in_ptr = dynamic_cast(ptr(Node, rn), [ptr](in_node))
      regentlib.assert(not isnull(wire.in_ptr), "picked an invalid random pointer")

      var out_node = 0
      if (100 * drand48() < conf.pct_wire_in_piece) or (conf.num_pieces == 1) then
        out_node = ptr_offset + [uint](drand48() * conf.nodes_per_piece)
      else
        -- pick a random other piece and a node from there
        var pp = [uint](drand48() * (conf.num_pieces - 1))
        if pp >= p then pp += 1 end

        -- pick an arbitrary node, except that if it's one that didn't used to be shared, make the
        -- sequentially next pointer shared instead so that each node's shared pointers stay compact
        var idx = [uint](drand48() * conf.nodes_per_piece)
        if idx > piece_shared_nodes[pp] then
          idx = piece_shared_nodes[pp]
          piece_shared_nodes[pp] = piece_shared_nodes[pp] + 1
        end
        out_node = pp * conf.nodes_per_piece + idx
      end
      wire.out_ptr = dynamic_cast(ptr(Node, rn), [ptr](out_node))
      regentlib.assert(not isnull(wire.out_ptr),
        "picked an invalid random pointer within a piece")
    end
  end
  c.free(piece_shared_nodes)
end

task init_pointers(rn : region(Node),
                   rw : region(Wire(rn)))
where reads(rn),
      reads writes(rw.{in_ptr, out_ptr})
do
  var invalid_pointers = 0
  for w in rw do
    w.in_ptr = dynamic_cast(ptr(Node, rn), w.in_ptr)
    if isnull(w.in_ptr) then invalid_pointers += 1 end
    w.out_ptr = dynamic_cast(ptr(Node, rn), w.out_ptr)
    if isnull(w.out_ptr) then invalid_pointers += 1 end
  end
  return invalid_pointers == 0
end

task calculate_new_currents(steps : uint,
                            rn : region(Node),
                            rw : region(Wire(rn)))
where reads(rn.node_voltage,
            rw.{in_ptr, out_ptr, inductance, resistance, wire_cap}),
      reads writes(rw.{current, voltage})
do
  var recip_dt : float = 1.0 / DT
  __demand(__vectorize)
  for w in rw do
    var temp_v : float[WIRE_SEGMENTS + 1]
    var temp_i : float[WIRE_SEGMENTS]
    var old_i : float[WIRE_SEGMENTS]
    var old_v : float[WIRE_SEGMENTS - 1]

    temp_i[0] = w.current._0
    temp_i[1] = w.current._1
    temp_i[2] = w.current._2
    for i = 0, WIRE_SEGMENTS do old_i[i] = temp_i[i] end

    temp_v[1] = w.voltage._0
    temp_v[2] = w.voltage._1
    for i = 0, WIRE_SEGMENTS - 1 do old_v[i] = temp_v[i + 1] end

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
        temp_v[i + 1] = old_v[i] + DT * (temp_i[i] - temp_i[i + 1]) * recip_capacitance
      end
    end

    -- Write out the results
    w.current._0 = temp_i[0]
    w.current._1 = temp_i[1]
    w.current._2 = temp_i[2]

    w.voltage._0 = temp_v[1]
    w.voltage._1 = temp_v[2]
  end
end

task distribute_charge(rn : region(Node),
                       rw : region(Wire(rn)))
where reads(rw.{in_ptr, out_ptr, current._0, current._2}),
      reads writes(rn.charge)
do
  for w in rw do
    var in_current = -DT * w.current._0
    var out_current = DT * w.current._2
    w.in_ptr.charge += in_current
    w.out_ptr.charge += out_current
  end
end

task update_voltages(rn : region(Node))
where
  reads(rn.{node_cap, leakage}),
  reads writes(rn.{node_voltage, charge})
do
  for node in rn do
    var voltage = node.node_voltage + node.charge / node.node_cap
    voltage = voltage * (1.0 - node.leakage)
    node.node_voltage = voltage
    node.charge = 0.0
  end
end

task block(rn : region(Node),
           rw : region(Wire(rn)))
where reads(rn, rw)
do
  return 1
end

terra wait_for(x : int)
  return x
end

task toplevel()
  var conf : Config
  conf.num_loops = 5
  conf.num_pieces = 4
  conf.nodes_per_piece = 2
  conf.wires_per_piece = 4
  conf.pct_wire_in_piece = 95
  conf.random_seed = 12345
  conf.steps = 10000

  conf = parse_input_args(conf)
  c.printf("circuit settings: loops=%d pieces=%d nodes/piece=%d wires/piece=%d pct_in_piece=%d seed=%d\n",
    conf.num_loops, conf.num_pieces, conf.nodes_per_piece, conf.wires_per_piece,
    conf.pct_wire_in_piece, conf.random_seed)

  var num_circuit_nodes = conf.num_pieces * conf.nodes_per_piece
  var num_circuit_wires = conf.num_pieces * conf.wires_per_piece

  var rn = region(ispace(ptr, num_circuit_nodes), Node)
  var rw = region(ispace(ptr, num_circuit_wires), Wire(rn))

  load_circuit(rn, rw, conf)

  var pointer_checks = init_pointers(rn, rw)
  regentlib.assert(pointer_checks, "there are some invalid pointers")

  wait_for(block(rn, rw))

  c.printf("Starting main simulation loop\n")
  var ts_start = c.legion_get_current_time_in_micros()
  var steps = conf.steps
  for j = 0, conf.num_loops do
    c.legion_runtime_begin_trace(__runtime(), __context(), 0)

    calculate_new_currents(steps, rn, rw)
    distribute_charge(rn, rw)
    update_voltages(rn)

    c.legion_runtime_end_trace(__runtime(), __context(), 0)
  end
  -- Force all previous tasks to complete before continuing.
  wait_for(block(rn, rw))
  var ts_end = c.legion_get_current_time_in_micros()

  do
    var sim_time = 1e-6 * (ts_end - ts_start)
    c.printf("ELAPSED TIME = %7.3f s\n", sim_time)

    -- Compute the floating point operations per second
    var num_circuit_nodes : uint64 = conf.num_pieces * conf.nodes_per_piece
    var num_circuit_wires : uint64 = conf.num_pieces * conf.wires_per_piece
    -- calculate currents
    var operations : uint64 =
      num_circuit_wires * (WIRE_SEGMENTS * 6 + (WIRE_SEGMENTS - 1) * 4) * conf.steps
    -- distribute charge
    operations = operations + (num_circuit_wires * 4)
    -- update voltages
    operations = operations + (num_circuit_nodes * 4)
    -- multiply by the number of loops
    operations = operations * conf.num_loops

    -- Compute the number of gflops
    var gflops = (1e-9 * operations) / sim_time
    c.printf("GFLOPS = %7.3f GFLOPS\n", gflops)
  end
  c.printf("simulation complete - destroying regions\n")
end
regentlib.start(toplevel)
