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
-- [
--   ["-ll:cpu", "4", "-ll:py", "1", "-ll:pyimport", "circuit_sparse"],
-- ]

import "regent"

-- Compile and link circuit_mapper.cc
local cmapper
local cconfig
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
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
  local max_dim = os.getenv('MAX_DIM') or '3'

  local cxx_flags = os.getenv('CC_FLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror -DLEGION_MAX_DIM=" .. max_dim .. " -DREALM_MAX_DIM=" .. max_dim
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
  cmapper = terralib.includec("circuit_mapper.h", {"-I", root_dir, "-I", runtime_dir})
  cconfig = terralib.includec("circuit_config.h", {"-I", root_dir, "-I", runtime_dir})
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

task init_piece(spiece_id   : int,
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

task calculate_new_currents(print_ts : bool,
                            steps : uint,
                            rpn : region(node),
                            rsn : region(node),
                            rgn : region(node),
                            rw : region(wire(rpn, rsn, rgn)))
where
  reads(rpn.node_voltage, rsn.node_voltage, rgn.node_voltage,
        rw.{in_ptr, out_ptr, inductance, resistance, wire_cap}),
  reads writes(rw.{current, voltage})
do
  if print_ts then
    c.printf("t: %ld\n", c.legion_get_current_time_in_micros())
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

task distribute_charge(rpn : region(node),
                       rsn : region(node),
                       rgn : region(node),
                       rw : region(wire(rpn, rsn, rgn)))
where
  reads(rw.{in_ptr, out_ptr, current._0, current._9}),
  reduces +(rpn.charge, rsn.charge, rgn.charge)
do
  var dt = DELTAT
  for w in rw do
    var in_current : float = -dt * w.current._0
    var out_current : float = dt * w.current._9
    w.in_ptr.charge += in_current
    w.out_ptr.charge += out_current
  end
end

task update_voltages(print_ts : bool,
                     rpn : region(node),
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
  if print_ts then
    c.printf("t: %ld\n", c.legion_get_current_time_in_micros())
  end
end

extern task main()
main:set_task_id(2)

if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
  local link_flags = terralib.newlist({"-L" .. out_dir, "-lcircuit_mapper", "-lm"})

  if os.getenv('STANDALONE') == '1' then
    os.execute('cp ' .. os.getenv('LG_RT_DIR') .. '/../bindings/regent/libregent.so ' .. out_dir)
  end

  local exe = os.getenv('OBJNAME') or "circuit"
  regentlib.saveobj(main, exe, "executable", cmapper.register_mappers, link_flags)
else
  regentlib.start(main, cmapper.register_mappers)
end
