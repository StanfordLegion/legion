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

import "regent"

local c = regentlib.c
local std = terralib.includec("stdlib.h")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)

local CktConfig = require("circuit_config")

local helper = {}

local WIRE_SEGMENTS = 3
local DT = 1e-6

task helper.generate_random_circuit(rn : region(Node),
                                    rw : region(Wire(rn, rn, rn)),
                                    conf : CktConfig)
where reads writes(rn, rw)
do
  var piece_shared_nodes : &uint =
    [&uint](c.malloc([sizeof(uint)] * conf.num_pieces))
  for i = 0, conf.num_pieces do piece_shared_nodes[i] = 0 end

  srand48(conf.random_seed)

  var npieces = conf.num_pieces
  var npp = conf.nodes_per_piece
  var wpp = conf.wires_per_piece

  for p = 0, npieces do
    for i = 0, npp do
      var node = dynamic_cast(ptr(Node, rn), ptr(p * npp + i))
      node.node_cap = drand48() + 1.0
      node.leakage = 0.1 * drand48()
      node.charge = 0.0
      node.node_voltage = 2 * drand48() - 1.0
    end
  end

  for p = 0, npieces do
    var ptr_offset = p * npp
    for i = 0, wpp do
      var wire = dynamic_cast(ptr(Wire(rn, rn, rn), rw), ptr(p * wpp + i))
      wire.current._0 = 0.0
      wire.current._1 = 0.0
      wire.current._2 = 0.0
      wire.voltage._0 = 0.0
      wire.voltage._1 = 0.0
      wire.resistance = drand48() * 10.0 + 1.0

      -- Keep inductance on the order of 1e-3 * dt to avoid resonance problems
      wire.inductance = (drand48() + 0.1) * DT * 1e-3
      wire.wire_cap = drand48() * 0.1

      var in_node = ptr_offset + [uint](drand48() * npp)
      wire.in_ptr = dynamic_cast(ptr(Node, rn, rn), ptr(in_node))
      regentlib.assert(not isnull(wire.in_ptr),
        "picked an invalid random pointer")

      var out_node = 0
      if (100 * drand48() < conf.pct_wire_in_piece) or (npieces == 1) then
        out_node = ptr_offset + [uint](drand48() * npp)
      else
        -- pick a random other piece and a node from there
        var pp = [uint](drand48() * (conf.num_pieces - 1))
        if pp >= p then pp += 1 end

        -- pick an arbitrary node, except that if it's one that didn't used to be shared, make the
        -- sequentially next pointer shared instead so that each node's shared pointers stay compact
        var idx = [uint](drand48() * npp)
        if idx > piece_shared_nodes[pp] then
          idx = piece_shared_nodes[pp]
          piece_shared_nodes[pp] = piece_shared_nodes[pp] + 1
        end
        out_node = pp * npp + idx
      end
      wire.out_ptr = dynamic_cast(ptr(Node, rn, rn, rn), ptr(out_node))
      regentlib.assert(not isnull(wire.out_ptr),
        "picked an invalid random pointer within a piece")
    end
  end
  c.free(piece_shared_nodes)
end

task helper.validate_pointers(rpn : region(Node),
                              rsn : region(Node),
                              rgn : region(Node),
                              rw : region(Wire(rpn, rsn, rgn)))
where reads(rpn, rsn, rgn),
      reads writes(rw.{in_ptr, out_ptr})
do
  for w in rw do
    w.in_ptr = dynamic_cast(ptr(Node, rpn, rsn), w.in_ptr)
    if isnull(w.in_ptr) then
      c.printf("validation error: wire %d does not originate from the piece it belongs\n",
        __raw(w))
      regentlib.assert(false, "pointer validation failed")
    end
    w.out_ptr = dynamic_cast(ptr(Node, rpn, rsn, rgn), w.out_ptr)
    if isnull(w.out_ptr) then
      c.printf("validation error: wire %d points to an invalid node\n",
        __raw(w))
      regentlib.assert(false, "pointer validation failed")
    end
  end
end

task helper.dump_graph(conf : CktConfig,
                       rn : region(Node),
                       rw : region(Wire(rn, rn, rn)),
                       pn_private : partition(disjoint, rn, ispace(int1d)),
                       pn_shared : partition(disjoint, rn, ispace(int1d)),
                       pn_ghost : partition(aliased, rn, ispace(int1d)),
                       pw_outgoing : partition(disjoint, rw, ispace(int1d)))
where reads(rn, rw)
do
  for i = 0, conf.num_pieces do
    var rpn = pn_private[i]
    var rsn = pn_shared[i]
    var rgn = pn_ghost[i]
    c.printf("piece %d:\n", i)
    for n in rpn do
      c.printf("  private node %d\n", __raw(n))
    end
    for n in rsn do
      c.printf("  shared node %d\n", __raw(n))
    end
    for n in rgn do
      c.printf("  ghost node %d\n", __raw(n))
    end
    for w in pw_outgoing[i] do
      c.printf("  outgoing edge %d: %d -> %d\n",
        __raw(w), __raw(w.in_ptr), __raw(w.out_ptr))
    end
  end
end


terra helper.calculate_gflops(sim_time : double, conf : CktConfig)

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

  var gflops = (1e-9 * operations) / sim_time
  return gflops
end

helper.timestamp = c.legion_get_current_time_in_micros

local task block(rn : region(Node), rw : region(Wire(rn, rn, rn)))
where reads(rn, rw)
do
  return 1
end

local terra wait_for(x : int)
end

__demand(__inline)
task helper.wait_for(rn : region(Node), rw : region(Wire(rn, rn, rn)))
where reads(rn, rw)
do
  wait_for(block(rn, rw))
end

return helper
