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
-- [["-hl:no_dyn"]]

import "regent"

local c = regentlib.c

WIRE_SEGMENTS = 3
DT = 1e-6

struct Currents {
  _0 : float,
  _1 : float,
  _2 : float,
}

struct Voltages {
  _0 : float,
  _1 : float,
}

fspace Node {
  node_cap      : float,
  leakage       : float,
  charge        : float,
  node_voltage  : float,
}

fspace Wire(rpn : region(Node),
            rsn : region(Node),
            rgn : region(Node)) {
  in_ptr     : ptr(Node, rpn, rsn),
  out_ptr    : ptr(Node, rpn, rsn,  rgn),
  inductance : float,
  resistance : float,
  wire_cap   : float,
  current    : Currents,
  voltage    : Voltages,
}

local CktConfig = require("circuit_config")
local helper = require("circuit_helper_dep")

task calculate_new_currents(steps : uint,
                            rpn : region(Node),
                            rsn : region(Node),
                            rgn : region(Node),
                            rw : region(Wire(rpn, rsn, rgn)))
where reads(rpn.node_voltage, rsn.node_voltage, rgn.node_voltage,
            rw.{in_ptr, out_ptr, inductance, resistance, wire_cap}),
      reads writes(rw.{current, voltage})
do
  var recip_dt : float = 1.0 / DT
  --__demand(__vectorize)
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

task distribute_charge(rpn : region(Node),
                       rsn : region(Node),
                       rgn : region(Node),
                       rw : region(Wire(rpn, rsn, rgn)))
where reads(rw.{in_ptr, out_ptr, current._0, current._2}),
      reduces +(rpn.charge, rsn.charge, rgn.charge)
do
  for w in rw do
    var in_current = -DT * w.current._0
    var out_current = DT * w.current._2
    w.in_ptr.charge += in_current
    w.out_ptr.charge += out_current
  end
end

task update_voltages(rn : region(Node))
where reads(rn.{node_cap, leakage}),
      reads writes(rn.{node_voltage, charge})
do
  for node in rn do
    var voltage = node.node_voltage + node.charge / node.node_cap
    voltage = voltage * (1.0 - node.leakage)
    node.node_voltage = voltage
    node.charge = 0.0
  end
end

terra f()
  var x : Wire(wild, wild, wild)
  return x.in_ptr
end
f:compile()

task toplevel()
  var conf : CktConfig
  conf:initialize_from_command()
  conf:show()

  var num_circuit_nodes = conf.num_pieces * conf.nodes_per_piece
  var num_circuit_wires = conf.num_pieces * conf.wires_per_piece

  var rn = region(ispace(ptr, num_circuit_nodes), Node)
  var rw = region(ispace(ptr, num_circuit_wires), Wire(wild, wild, wild))

  c.printf("Generating random circuit...\n")
  helper.generate_random_circuit(rn, rw, conf)

  c.printf("Creating partitions...\n")
  var colors = ispace(int1d, conf.num_pieces)
  var pn_equal = partition(equal, rn, colors)
  var pw_out = preimage(rw, pn_equal, rw.in_ptr)
  var pw_in = preimage(rw, pn_equal, rw.out_ptr)
  var pn_out = image(rn, pw_out, rw.out_ptr)
  var pn_in = image(rn, pw_in, rw.in_ptr)
  var pn_private = pn_equal & pn_in & pn_out
  var pn_shared = pn_equal - pn_private
  var pn_ghost = pn_out - pn_equal

  if conf.dump_graph then
    helper.dump_graph(conf, rn, rw,
                      pn_private, pn_shared,
                      pn_ghost, pw_out)
  end

  __demand(__parallel)
  for i = 0, conf.num_pieces do
    helper.validate_pointers(pn_private[i],
                             pn_shared[i],
                             pn_ghost[i],
                             pw_out[i])
  end

  helper.wait_for(rn, rw)

  c.printf("Starting main simulation loop\n")
  var ts_start = c.legion_get_current_time_in_micros()
  var steps = conf.steps
  for j = 0, conf.num_loops do
    c.legion_runtime_begin_trace(__runtime(), __context(), 0, false)

    __demand(__parallel)
    for i = 0, conf.num_pieces do
      calculate_new_currents(steps,
                             pn_private[i],
                             pn_shared[i],
                             pn_ghost[i],
                             pw_out[i])
    end
    __demand(__parallel)
    for i = 0, conf.num_pieces do
      distribute_charge(pn_private[i],
                        pn_shared[i],
                        pn_ghost[i],
                        pw_out[i])
    end
    __demand(__parallel)
    for i = 0, conf.num_pieces do
      update_voltages(pn_equal[i])
    end

    c.legion_runtime_end_trace(__runtime(), __context(), 0)
  end
  -- Force all previous tasks to complete before continuing.
  helper.wait_for(rn, rw)
  var ts_end = c.legion_get_current_time_in_micros()

  var sim_time = 1e-6 * (ts_end - ts_start)
  c.printf("ELAPSED TIME = %7.3f s\n", sim_time)

  -- Compute the number of gflops
  var gflops = helper.calculate_gflops(sim_time, conf)
  c.printf("GFLOPS = %7.3f GFLOPS\n", gflops)
  c.printf("simulation complete\n")
end
regentlib.start(toplevel)
