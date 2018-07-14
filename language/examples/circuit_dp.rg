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

import "regent"

-- Compile and link circuit.cc
local ccircuit
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = root_dir .. "../../runtime"
  local circuit_cc = root_dir .. "circuit.cc"
  local circuit_so = os.tmpname() .. ".so" -- root_dir .. "circuit.so"
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
                 circuit_cc .. " -o " .. circuit_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. circuit_cc)
    assert(false)
  end
  terralib.linklibrary(circuit_so)
  ccircuit = terralib.includec("circuit.h", {"-I", root_dir, "-I", runtime_dir})
end

local c = regentlib.c
local std = terralib.includec("stdlib.h")
local cstring = terralib.includec("string.h")
rawset(_G, "drand48", std.drand48)
rawset(_G, "srand48", std.srand48)

WIRE_SEGMENTS = 10
STEPS = 10000
DELTAT = 1e-6

struct Colorings {
  privacy_map : c.legion_coloring_t,
  private_node_map : c.legion_coloring_t,
  shared_node_map : c.legion_coloring_t,
  ghost_node_map : c.legion_coloring_t,
  wire_owner_map : c.legion_coloring_t,
  first_wires : &c.legion_ptr_t,
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
  dump_values : bool
  use_dense_kernel : bool,
}

fspace node {
  node_cap : float,
  leakage  : float,
  charge   : float,
  node_voltage  : float,
  subckt_id : int,
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

fspace wire(rpn : region(node),
            rsn : region(node),
            rgn : region(node)) {
  in_ptr : ptr(node, rpn, rsn, rgn),
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
    elseif cstring.strcmp(args.argv[i], "-dense") == 0 then
      conf.use_dense_kernel = true
    end
  end
  return conf
end

terra random_element(arr : &c.legion_ptr_t,
                     num_elmts : uint)
  var index = [uint](drand48() * num_elmts)
  return arr[index]
end

-- NOTE: assumes array is sorted!
terra contains_element(arr : &c.legion_ptr_t,
		       num_elmts : uint,
		       needle : c.legion_ptr_t)
  --for i = 0, num_elmts - 1 do
  --   regentlib.assert(arr[i].value < arr[i + 1].value, "not sorted")
  --end
  --var slow_find = -1
  --for i = 0, num_elmts do
  --   if arr[i].value == needle.value then slow_find = i end
  --end
  var lo = 0
  var hi = num_elmts
  while lo < hi do
    var mid = (lo + hi) >> 1
    var v = arr[mid].value
    if v < needle.value then
      lo = mid + 1
    elseif v > needle.value then
      hi = mid
    else
      --regentlib.assert(slow_find >= 0, "mismatch1")
      return true
    end
  end
  --regentlib.assert(slow_find == -1, "mismatch2")
  return false
end

terra load_circuit(runtime : c.legion_runtime_t,
                   ctx : c.legion_context_t,
                   all_nodes : c.legion_physical_region_t[5],
                   all_nodes_fields : c.legion_field_id_t[5],
                   all_wires : c.legion_physical_region_t[26],
                   all_wires_fields : c.legion_field_id_t[26],
                   conf : Config)


  var piece_shared_nodes = [&uint](c.malloc(conf.num_pieces * sizeof(uint)))

  srand48(conf.random_seed)

  var fa_node_cap =
    c.legion_physical_region_get_field_accessor_array_1d(all_nodes[0], all_nodes_fields[0])
  var fa_node_leakage =
    c.legion_physical_region_get_field_accessor_array_1d(all_nodes[1], all_nodes_fields[1])
  var fa_node_charge =
    c.legion_physical_region_get_field_accessor_array_1d(all_nodes[2], all_nodes_fields[2])
  var fa_node_voltage =
    c.legion_physical_region_get_field_accessor_array_1d(all_nodes[3], all_nodes_fields[3])
  var fa_node_subckt =
    c.legion_physical_region_get_field_accessor_array_1d(all_nodes[4], all_nodes_fields[4])
  var all_nodes_ispace = c.legion_physical_region_get_logical_region(all_nodes[0]).index_space
  --var first_nodes : &c.legion_ptr_t =
  --  [&c.legion_ptr_t](c.malloc(conf.num_pieces * sizeof(c.legion_ptr_t)))
  var piece_node_ptrs : &&c.legion_ptr_t =
    [&&c.legion_ptr_t](c.malloc(conf.num_pieces * 8))
  do
    var itr = c.legion_index_iterator_create(runtime, ctx, all_nodes_ispace)
    for n = 0, conf.num_pieces do
      piece_node_ptrs[n] =
        [&c.legion_ptr_t](c.malloc(conf.nodes_per_piece * sizeof(c.legion_ptr_t)))
      for i = 0, conf.nodes_per_piece do
        regentlib.assert(c.legion_index_iterator_has_next(itr), "test failed")
        var node_ptr = c.legion_index_iterator_next(itr)
        --if i == 0 then
        --  first_nodes[n] = node_ptr
        --end
        @[&int](c.legion_accessor_array_1d_ref(fa_node_subckt, node_ptr)) = n
        var capacitance = drand48() + 1.0
        @[&float](c.legion_accessor_array_1d_ref(fa_node_cap, node_ptr)) = capacitance
        var leakage = 0.1 * drand48()
        @[&float](c.legion_accessor_array_1d_ref(fa_node_leakage, node_ptr)) = leakage
        @[&float](c.legion_accessor_array_1d_ref(fa_node_charge, node_ptr)) = 0.0
        var init_voltage = 2 * drand48() - 1.0
        @[&float](c.legion_accessor_array_1d_ref(fa_node_voltage, node_ptr)) = init_voltage
        piece_node_ptrs[n][i] = node_ptr
      end
    end
    c.legion_index_iterator_destroy(itr)
  end
  c.legion_accessor_array_1d_destroy(fa_node_cap)
  c.legion_accessor_array_1d_destroy(fa_node_leakage)
  c.legion_accessor_array_1d_destroy(fa_node_charge)
  c.legion_accessor_array_1d_destroy(fa_node_voltage)
  c.legion_accessor_array_1d_destroy(fa_node_subckt)

  var fa_wire_in_ptr =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[0], all_wires_fields[0])
  var fa_wire_in_ptr_index =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[1], all_wires_fields[1])
  var fa_wire_out_ptr =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[2], all_wires_fields[2])
  var fa_wire_out_ptr_index =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[3], all_wires_fields[3])
  var fa_wire_inductance =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[4], all_wires_fields[4])
  var fa_wire_resistance =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[5], all_wires_fields[5])
  var fa_wire_cap =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[6], all_wires_fields[6])
  var fa_wire_currents : &c.legion_accessor_array_1d_t =
    [&c.legion_accessor_array_1d_t](c.malloc(sizeof(c.legion_accessor_array_1d_t) * WIRE_SEGMENTS))
  for i = 0, WIRE_SEGMENTS do
    fa_wire_currents[i] =
      c.legion_physical_region_get_field_accessor_array_1d(all_wires[7 + i], all_wires_fields[7 + i])
  end
  var fa_wire_voltages =
    [&c.legion_accessor_array_1d_t](c.malloc(sizeof(c.legion_accessor_array_1d_t) * (WIRE_SEGMENTS - 1)))
  for i = 0, WIRE_SEGMENTS - 1 do
    fa_wire_voltages[i] =
      c.legion_physical_region_get_field_accessor_array_1d(all_wires[17 + i], all_wires_fields[17 + i])
  end
  var all_wires_ispace = c.legion_physical_region_get_logical_region(all_wires[0]).index_space
  do
    var itr = c.legion_index_iterator_create(runtime, ctx, all_wires_ispace)
    for n = 0, conf.num_pieces do
      for i = 0, conf.wires_per_piece do
        regentlib.assert(c.legion_index_iterator_has_next(itr), "test failed")
        var wire_ptr = c.legion_index_iterator_next(itr)
        for j = 0, WIRE_SEGMENTS do
          @[&float](c.legion_accessor_array_1d_ref(fa_wire_currents[j], wire_ptr)) = 0.0
        end
        for j = 0, WIRE_SEGMENTS - 1 do
          @[&float](c.legion_accessor_array_1d_ref(fa_wire_voltages[j], wire_ptr)) = 0.0
        end

        var resistance = drand48() * 10.0 + 1.0
        @[&float](c.legion_accessor_array_1d_ref(fa_wire_resistance, wire_ptr)) = resistance
        -- Keep inductance on the order of 1e-3 * dt to avoid resonance problems
        var inductance = (drand48() + 0.1) * DELTAT * 1e-3
        @[&float](c.legion_accessor_array_1d_ref(fa_wire_inductance, wire_ptr)) = inductance
        var capacitance = drand48() * 0.1
        @[&float](c.legion_accessor_array_1d_ref(fa_wire_cap, wire_ptr)) = capacitance

        @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(fa_wire_in_ptr, wire_ptr)) =
          random_element(piece_node_ptrs[n], conf.nodes_per_piece)

        if ((100 * drand48()) < conf.pct_wire_in_piece) or (conf.num_pieces == 1) then
          @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(fa_wire_out_ptr, wire_ptr)) =
            random_element(piece_node_ptrs[n], conf.nodes_per_piece)
        else
          -- pick a random other piece and a node from there
          var nn = [uint](drand48() * (conf.num_pieces - 1))
          if nn >= n then nn = nn + 1 end

          -- pick an arbitrary node, except that if it's one that didn't used to be shared, make the
          --  sequentially next pointer shared instead so that each node's shared pointers stay compact
          var idx = [uint](drand48() * conf.nodes_per_piece)
          if idx > piece_shared_nodes[nn] then
            idx = piece_shared_nodes[nn]
            piece_shared_nodes[nn] = piece_shared_nodes[nn] + 1
          end
          var out_ptr = piece_node_ptrs[nn][idx]

          @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(fa_wire_out_ptr, wire_ptr)) = out_ptr
        end
        @[&uint32](c.legion_accessor_array_1d_ref(fa_wire_in_ptr_index, wire_ptr)) = 0
        @[&uint32](c.legion_accessor_array_1d_ref(fa_wire_out_ptr_index, wire_ptr)) = 0
      end
    end
    c.legion_index_iterator_destroy(itr)
  end
  c.legion_accessor_array_1d_destroy(fa_wire_in_ptr)
  c.legion_accessor_array_1d_destroy(fa_wire_out_ptr)
  c.legion_accessor_array_1d_destroy(fa_wire_inductance)
  c.legion_accessor_array_1d_destroy(fa_wire_resistance)
  c.legion_accessor_array_1d_destroy(fa_wire_cap)
  for i = 0, WIRE_SEGMENTS do
    c.legion_accessor_array_1d_destroy(fa_wire_currents[i])
  end
  for i = 0, WIRE_SEGMENTS - 1 do
    c.legion_accessor_array_1d_destroy(fa_wire_voltages[i])
  end

  c.free(fa_wire_currents)
  c.free(fa_wire_voltages)
  c.free(piece_shared_nodes)
  for n = 0, conf.num_pieces do
    c.free(piece_node_ptrs[n])
  end
  c.free(piece_node_ptrs)
end

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

terra NOW(s : &int8)
  c.printf("now: %s = %lld\n", s, c.legion_get_current_time_in_micros())
end

terra create_colorings(runtime : c.legion_runtime_t,
                       ctx : c.legion_context_t,
		       all_nodes : c.legion_physical_region_t[5],
		       all_nodes_fields : c.legion_field_id_t[5],
		       all_wires : c.legion_physical_region_t[26],
		       all_wires_fields : c.legion_field_id_t[26],
		       conf : Config)

  var colorings : Colorings

  var wire_owner_map = c.legion_coloring_create()
  var private_node_map = c.legion_coloring_create()
  var shared_node_map = c.legion_coloring_create()
  var ghost_node_map = c.legion_coloring_create()
  var locator_node_map = c.legion_coloring_create()

  var privacy_map = c.legion_coloring_create()
  c.legion_coloring_ensure_color(privacy_map, 0)
  c.legion_coloring_ensure_color(privacy_map, 1)

  -- figure out which subckt each node says it's in
  var fa_node_subckt =
    c.legion_physical_region_get_field_accessor_array_1d(all_nodes[4], all_nodes_fields[4])

  var cfa_node_subckt = cache_accessor(fa_node_subckt)

  var piece_node_ptrs : &&c.legion_ptr_t =
    [&&c.legion_ptr_t](c.malloc(conf.num_pieces * 8))

  var all_nodes_ispace = c.legion_physical_region_get_logical_region(all_nodes[0]).index_space

  do
    var itr = c.legion_index_iterator_create(runtime, ctx, all_nodes_ispace)
    for n = 0, conf.num_pieces do
      c.legion_coloring_ensure_color(private_node_map, n)
      piece_node_ptrs[n] =
        [&c.legion_ptr_t](c.malloc(conf.nodes_per_piece * sizeof(c.legion_ptr_t)))
      for i = 0, conf.nodes_per_piece do
        regentlib.assert(c.legion_index_iterator_has_next(itr), "test failed")
        var node_ptr = c.legion_index_iterator_next(itr)

        --var subckt = @[&int](c.legion_accessor_array_1d_ref(fa_node_subckt, node_ptr))
        var subckt = @[&int](cfa_node_subckt:ref(node_ptr))
	regentlib.assert(subckt == n, "subckt mismatch")

        c.legion_coloring_add_point(private_node_map, n, node_ptr)
        c.legion_coloring_add_point(privacy_map, 0, node_ptr)
        piece_node_ptrs[n][i] = node_ptr
      end
    end
    c.legion_index_iterator_destroy(itr)
  end
  c.legion_accessor_array_1d_destroy(fa_node_subckt)

  -- now iterate over wires to (re)figure out topology
  var fa_wire_in_ptr =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[0], all_wires_fields[0])
  var fa_wire_out_ptr =
    c.legion_physical_region_get_field_accessor_array_1d(all_wires[2], all_wires_fields[2])

  var cfa_wire_in_ptr = cache_accessor(fa_wire_in_ptr)
  var cfa_wire_out_ptr = cache_accessor(fa_wire_out_ptr)

  var first_wires : &c.legion_ptr_t =
    [&c.legion_ptr_t](c.malloc(conf.num_pieces * sizeof(c.legion_ptr_t)))

  var all_wires_ispace = c.legion_physical_region_get_logical_region(all_wires[0]).index_space

  do
    var itr = c.legion_index_iterator_create(runtime, ctx, all_wires_ispace)
    for n = 0, conf.num_pieces do
      c.legion_coloring_ensure_color(ghost_node_map, n)
      var i = 0
      while i < conf.wires_per_piece do
	var req_count : c.size_t = conf.wires_per_piece - i
	var act_count : uint64 = 0
        var wire_ptr = c.legion_index_iterator_next_span(itr, &act_count, req_count)
	--c.printf("got %d + %d values\n", wire_ptr.value, act_count)
	for j = 0, act_count do
	  if i == 0 then
	    first_wires[n] = wire_ptr
          end

	  --var in_ptr = @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(fa_wire_in_ptr, wire_ptr))
	  var in_ptr = @[&c.legion_ptr_t](cfa_wire_in_ptr:ref(wire_ptr))
	  --var out_ptr = @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(fa_wire_out_ptr, wire_ptr))
	  var out_ptr = @[&c.legion_ptr_t](cfa_wire_out_ptr:ref(wire_ptr))

	  var in_local = contains_element(piece_node_ptrs[n], conf.nodes_per_piece, in_ptr)
	  regentlib.assert(in_local, "in_ptr not local?")

	  var nn = n
	  var out_local = contains_element(piece_node_ptrs[n], conf.nodes_per_piece, out_ptr)
	  if not out_local then
	    for j = 0, conf.num_pieces do
	      if contains_element(piece_node_ptrs[j], conf.nodes_per_piece, out_ptr) then
		nn = j
		break
	      end
	    end
	    -- This node is no longer private
	    c.legion_coloring_delete_point(privacy_map, 0, out_ptr)
	    c.legion_coloring_add_point(privacy_map, 1, out_ptr)
	    c.legion_coloring_add_point(ghost_node_map, n, out_ptr)
	  end
	  c.legion_coloring_add_point(wire_owner_map, n, wire_ptr)
          i = i + 1
	  wire_ptr.value = wire_ptr.value + 1
        end
      end
    end
    c.legion_index_iterator_destroy(itr)
  end
  c.legion_accessor_array_1d_destroy(fa_wire_in_ptr)
  c.legion_accessor_array_1d_destroy(fa_wire_out_ptr)

  -- Second pass: make some random fraction of the private nodes shared
  do
    var itr = c.legion_index_iterator_create(runtime, ctx, all_nodes_ispace)
    for n = 0, conf.num_pieces do
      c.legion_coloring_ensure_color(shared_node_map, n)
      for i = 0, conf.nodes_per_piece do
        regentlib.assert(c.legion_index_iterator_has_next(itr), "test failed")
        var node_ptr = c.legion_index_iterator_next(itr)
        if not c.legion_coloring_has_point(privacy_map, 0, node_ptr) then
          c.legion_coloring_delete_point(private_node_map, n, node_ptr)
          c.legion_coloring_add_point(shared_node_map, n, node_ptr)
        end
      end
    end
    c.legion_index_iterator_destroy(itr)
  end


  colorings.privacy_map = privacy_map
  colorings.private_node_map = private_node_map
  colorings.shared_node_map = shared_node_map
  colorings.ghost_node_map = ghost_node_map
  colorings.wire_owner_map = wire_owner_map
  colorings.first_wires = first_wires

  for n = 0, conf.num_pieces do
    c.free(piece_node_ptrs[n])
  end
  c.free(piece_node_ptrs)

  return colorings
end
create_colorings:compile()

task init_pointers(rpn : region(node),
                   rsn : region(node),
                   rgn : region(node),
                   rw : region(wire(rpn, rsn, rgn)))
where
  reads(rpn, rsn, rgn),
  reads writes(rw.{in_ptr, out_ptr})
do
  for w in rw do
    w.in_ptr = dynamic_cast(ptr(node, rpn, rsn, rgn), w.in_ptr)
    w.out_ptr = dynamic_cast(ptr(node, rpn, rsn, rgn), w.out_ptr)
  end
  return 1
end

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

local vec4 = vector(float, 4)

local terra get_vec_node_voltage(current_wire : c.legion_ptr_t,
                                 priv  : c.legion_accessor_array_1d_t,
                                 shr   : c.legion_accessor_array_1d_t,
                                 ghost : c.legion_accessor_array_1d_t,
                                 ptrs  : c.legion_accessor_array_1d_t,
                                 locs  : c.legion_accessor_array_1d_t) : vec4
  var voltages : float[4]
  for i = 0, 4 do
    var ptr = c.legion_ptr_t { value = current_wire.value + i }
    var node_ptr =
      @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(ptrs, ptr))
    var loc =
      @[&uint](c.legion_accessor_array_1d_ref(locs, ptr))
    if loc == 1 then
      voltages[i] = @[&float](c.legion_accessor_array_1d_ref(priv, node_ptr))
    elseif loc == 2 then
      voltages[i] = @[&float](c.legion_accessor_array_1d_ref(shr, node_ptr))
    elseif loc == 3 then
      voltages[i] = @[&float](c.legion_accessor_array_1d_ref(ghost, node_ptr))
    else
      regentlib.assert(false, "bad location")
    end
  end
  return vector(voltages[0], voltages[1], voltages[2], voltages[3])
end

local terra get_node_voltage(priv  : c.legion_accessor_array_1d_t,
                             shr   : c.legion_accessor_array_1d_t,
                             ghost : c.legion_accessor_array_1d_t,
                             loc : uint,
                             ptr : c.legion_ptr_t)
  var tmp : float
  if loc == 1 then
    c.legion_accessor_array_1d_read(priv, ptr, &tmp, sizeof(float))
    return tmp
  elseif loc == 2 then
    c.legion_accessor_array_1d_read(shr, ptr, &tmp, sizeof(float))
    return tmp
  elseif loc == 3 then
    c.legion_accessor_array_1d_read(ghost, ptr, &tmp, sizeof(float))
    return tmp
  else
    regentlib.assert(false, "bad location")
  end
end

terra dense_calc_new_currents(steps : uint,
                              dt : float,
                              first_wire : c.legion_ptr_t,
                              num_wires : uint,
                              fa_in_ptr        : c.legion_accessor_array_1d_t,
                              fa_out_ptr       : c.legion_accessor_array_1d_t,
                              fa_in_loc        : c.legion_accessor_array_1d_t,
                              fa_out_loc       : c.legion_accessor_array_1d_t,
                              fa_inductance    : c.legion_accessor_array_1d_t,
                              fa_resistance    : c.legion_accessor_array_1d_t,
                              fa_wire_cap      : c.legion_accessor_array_1d_t,
                              fa_pvt_voltage   : c.legion_accessor_array_1d_t,
                              fa_shr_voltage   : c.legion_accessor_array_1d_t,
                              fa_ghost_voltage : c.legion_accessor_array_1d_t,
                              fa_current       : &c.legion_accessor_array_1d_t,
                              fa_voltage       : &c.legion_accessor_array_1d_t)
  var index      : uint = 0
  var temp_v     : vec4[WIRE_SEGMENTS+1]
  var temp_i     : vec4[WIRE_SEGMENTS+1]
  var old_i      : vec4[WIRE_SEGMENTS]
  var old_v      : vec4[WIRE_SEGMENTS-1]
  var recip_dt   : vec4  = 1.0 / dt

  while index + 3 < num_wires do
    var current_wire : c.legion_ptr_t = c.legion_ptr_t { value = first_wire.value + index }
    for i = 0, WIRE_SEGMENTS do
      temp_i[i] = @[&vec4](c.legion_accessor_array_1d_ref(fa_current[i], current_wire))
      old_i[i] = temp_i[i]
    end
    for i = 0, WIRE_SEGMENTS - 1 do
      temp_v[i + 1] = @[&vec4](c.legion_accessor_array_1d_ref(fa_voltage[i], current_wire))
      old_v[i] = temp_v[i + 1]
    end
    temp_v[0] = get_vec_node_voltage(current_wire, fa_pvt_voltage,
                                     fa_shr_voltage, fa_ghost_voltage,
                                     fa_in_ptr, fa_in_loc)
    temp_v[WIRE_SEGMENTS] = get_vec_node_voltage(current_wire, fa_pvt_voltage,
                                                 fa_shr_voltage, fa_ghost_voltage,
                                                 fa_out_ptr, fa_out_loc)
    var inductance : vec4 = @[&vec4](c.legion_accessor_array_1d_ref(fa_inductance, current_wire))
    var recip_resistance  : vec4 = 1.0 / @[&vec4](c.legion_accessor_array_1d_ref(fa_resistance, current_wire))
    var recip_capacitance : vec4 = 1.0 / @[&vec4](c.legion_accessor_array_1d_ref(fa_wire_cap, current_wire))
    for j = 0, steps do
      for i = 0, WIRE_SEGMENTS do
        var dv  : vec4 = temp_v[i + 1] - temp_v[i]
        var di  : vec4 = temp_i[i] - old_i[i]
        var vol : vec4 = dv - inductance * di * recip_dt
        temp_i[i] = vol * recip_resistance
      end
      for i = 0, WIRE_SEGMENTS - 1 do
        var dq  : vec4 = dt * (temp_i[i] - temp_i[i + 1])
        temp_v[i + 1] = old_v[i] + dq * recip_capacitance
      end
    end

    for i = 0, WIRE_SEGMENTS do
      @[&vec4](c.legion_accessor_array_1d_ref(fa_current[i], current_wire)) = temp_i[i]
    end
    for i = 0, WIRE_SEGMENTS - 1 do
      @[&vec4](c.legion_accessor_array_1d_ref(fa_voltage[i], current_wire)) = temp_v[i + 1]
    end
    index = index + 4
  end

  while index < num_wires do
    var temp_v : float[WIRE_SEGMENTS + 1]
    var temp_i : float[WIRE_SEGMENTS]
    var old_i  : float[WIRE_SEGMENTS]
    var old_v  : float[WIRE_SEGMENTS - 1]
    var wire_ptr : c.legion_ptr_t = c.legion_ptr_t { value = first_wire.value + index }

    for i = 0, WIRE_SEGMENTS do
      temp_i[i] = @[&float](c.legion_accessor_array_1d_ref(fa_current[i], wire_ptr))
      old_i[i] = temp_i[i]
    end
    for i = 0, WIRE_SEGMENTS - 1 do
      temp_v[i + 1] = @[&float](c.legion_accessor_array_1d_ref(fa_voltage[i], wire_ptr))
      old_v[i] = temp_v[i+1]
    end

    -- Pin the outer voltages to the node voltages
    var in_ptr = @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(fa_in_ptr, wire_ptr))
    var in_loc = @[&uint](c.legion_accessor_array_1d_ref(fa_in_loc, wire_ptr))
    temp_v[0] =
      get_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, in_loc, in_ptr)
    var out_ptr = @[&c.legion_ptr_t](c.legion_accessor_array_1d_ref(fa_out_ptr, wire_ptr))
    var out_loc = @[&uint](c.legion_accessor_array_1d_ref(fa_out_loc, wire_ptr))
    temp_v[WIRE_SEGMENTS] =
      get_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, out_loc, out_ptr)

    -- Solve the RLC model iteratively
    var inductance : float = @[&float](c.legion_accessor_array_1d_ref(fa_inductance, wire_ptr))
    var recip_resistance : float = 1.0 / @[&float](c.legion_accessor_array_1d_ref(fa_resistance, wire_ptr))
    var recip_capacitance : float = 1.0 / @[&float](c.legion_accessor_array_1d_ref(fa_wire_cap, wire_ptr))
    var recip_dt : float = 1.0 / dt
    for j = 0, steps do
      -- first, figure out the new current from the voltage differential
      -- and our inductance:
      -- dV = R*I + L*I' ==> I = (dV - L*I')/R
      for i = 0, WIRE_SEGMENTS do
        temp_i[i] = ((temp_v[i+1] - temp_v[i]) -
                     (inductance * (temp_i[i] - old_i[i]) * recip_dt)) * recip_resistance
      end
      -- Now update the inter-node voltages
      for i = 0, WIRE_SEGMENTS - 1 do
        temp_v[i+1] = old_v[i] + dt * (temp_i[i] - temp_i[i+1]) * recip_capacitance
      end
    end

    -- Write out the results
    for i = 0, WIRE_SEGMENTS do
      @[&float](c.legion_accessor_array_1d_ref(fa_current[i], wire_ptr)) = temp_i[i]
    end
    for i = 0, WIRE_SEGMENTS - 1 do
      @[&float](c.legion_accessor_array_1d_ref(fa_voltage[i], wire_ptr)) = temp_v[i + 1]
    end
    -- Update the index
    index = index + 1
  end
end

task dense_calculate_new_currents(steps : uint,
                                  dt : float,
                                  first_wire : c.legion_ptr_t,
                                  num_wires : uint,
                                  rpn : region(node),
                                  rsn : region(node),
                                  rgn : region(node),
                                  rw : region(wire(rpn, rsn, rgn)))
where
  reads(rpn.node_voltage, rsn.node_voltage, rgn.node_voltage,
        rw.{in_ptr, out_ptr, inductance, resistance, wire_cap}),
  reads writes(rw.{current, voltage})
do
  var phy_rpn = __physical(rpn)
  var phy_rsn = __physical(rsn)
  var phy_rgn = __physical(rgn)
  var phy_rw = __physical(rw)

  var fid_rpn = __fields(rpn)
  var fid_rsn = __fields(rsn)
  var fid_rgn = __fields(rgn)
  var fid_rw = __fields(rw)

  var fa_current : c.legion_accessor_array_1d_t[WIRE_SEGMENTS]
  for i = 0, WIRE_SEGMENTS do
    fa_current[i] =
      c.legion_physical_region_get_field_accessor_array_1d(phy_rw[7 + i], fid_rw[7 + i])
  end
  var fa_voltage : c.legion_accessor_array_1d_t[WIRE_SEGMENTS - 1]
  for i = 0, WIRE_SEGMENTS - 1 do
    fa_voltage[i] =
      c.legion_physical_region_get_field_accessor_array_1d(phy_rw[17 + i], fid_rw[17 + i])
  end
  var fa_in_ptr =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rw[0], fid_rw[0])
  var fa_out_ptr =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rw[2], fid_rw[2])
  var fa_in_loc =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rw[1], fid_rw[1])
  var fa_out_loc =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rw[3], fid_rw[3])
  var fa_inductance =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rw[4], fid_rw[4])
  var fa_resistance =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rw[5], fid_rw[5])
  var fa_wire_cap =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rw[6], fid_rw[6])
  var fa_pvt_voltage =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rpn[0], fid_rpn[0])
  var fa_shr_voltage =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rsn[0], fid_rsn[0])
  var fa_ghost_voltage =
    c.legion_physical_region_get_field_accessor_array_1d(phy_rgn[0], fid_rgn[0])

  dense_calc_new_currents(steps, dt, first_wire, num_wires,
                          fa_in_ptr, fa_out_ptr, fa_in_loc, fa_out_loc,
                          fa_inductance, fa_resistance, fa_wire_cap,
                          fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage,
                          fa_current, fa_voltage)

  for i = 0, WIRE_SEGMENTS do
    c.legion_accessor_array_1d_destroy(fa_current[i])
  end
  for i = 0, WIRE_SEGMENTS - 1 do
    c.legion_accessor_array_1d_destroy(fa_voltage[i])
  end
  c.legion_accessor_array_1d_destroy(fa_in_ptr)
  c.legion_accessor_array_1d_destroy(fa_out_ptr)
  c.legion_accessor_array_1d_destroy(fa_in_loc)
  c.legion_accessor_array_1d_destroy(fa_out_loc)
  c.legion_accessor_array_1d_destroy(fa_inductance)
  c.legion_accessor_array_1d_destroy(fa_resistance)
  c.legion_accessor_array_1d_destroy(fa_wire_cap)
  c.legion_accessor_array_1d_destroy(fa_pvt_voltage)
  c.legion_accessor_array_1d_destroy(fa_shr_voltage)
  c.legion_accessor_array_1d_destroy(fa_ghost_voltage)
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

task dummy(rpn : region(node),
           rsn : region(node),
           rgn : region(node),
           rw : region(wire(rpn, rsn, rgn)))
where reads(rpn, rw) do
  return 1
end

terra wait_for(x : int)
  return x
end

task dump_task(rpn : region(node),
               rsn : region(node),
               rgn : region(node),
               rw : region(wire(rpn, rsn, rgn)))
where
  reads(rpn, rsn, rgn, rw)
do
  for w in rw do
    c.printf(" %.5g", w.current._0);
    c.printf(" %.5g", w.current._1);
    c.printf(" %.5g", w.current._2);
    c.printf(" %.5g", w.current._3);
    c.printf(" %.5g", w.current._4);
    c.printf(" %.5g", w.current._5);
    c.printf(" %.5g", w.current._6);
    c.printf(" %.5g", w.current._7);
    c.printf(" %.5g", w.current._8);
    c.printf(" %.5g", w.current._9);

    c.printf(" %.5g", w.voltage._0);
    c.printf(" %.5g", w.voltage._1);
    c.printf(" %.5g", w.voltage._2);
    c.printf(" %.5g", w.voltage._3);
    c.printf(" %.5g", w.voltage._4);
    c.printf(" %.5g", w.voltage._5);
    c.printf(" %.5g", w.voltage._6);
    c.printf(" %.5g", w.voltage._7);
    c.printf(" %.5g", w.voltage._8);

    c.printf("\n");
  end
end
task toplevel()
  var conf : Config
  conf.num_loops = 2
  conf.num_pieces = 4
  conf.nodes_per_piece = 2
  conf.wires_per_piece = 4
  conf.pct_wire_in_piece = 95
  conf.random_seed = 12345
  conf.steps = STEPS
  conf.sync = 0
  conf.perform_checks = false
  conf.dump_values = false
  conf.use_dense_kernel = false

  conf = parse_input_args(conf)
  c.printf("circuit settings: loops=%d pieces=%d nodes/piece=%d wires/piece=%d pct_in_piece=%d seed=%d\n",
    conf.num_loops, conf.num_pieces, conf.nodes_per_piece, conf.wires_per_piece,
    conf.pct_wire_in_piece, conf.random_seed)

  var num_circuit_nodes = conf.num_pieces * conf.nodes_per_piece
  var num_circuit_wires = conf.num_pieces * conf.wires_per_piece

  var all_nodes = region(ispace(ptr, num_circuit_nodes), node)
  var all_wires = region(ispace(ptr, num_circuit_wires), wire(wild, wild, wild))

  -- report mesh size in bytes
  do
    var node_size = [ terralib.sizeof(node) ]
    var wire_size = [ terralib.sizeof(wire(wild,wild,wild)) ]
    c.printf("Circuit memory usage:\n")
    c.printf("  Nodes : %9lld * %4d bytes = %11lld bytes\n", num_circuit_nodes, node_size, num_circuit_nodes * node_size)
    c.printf("  Wires : %9lld * %4d bytes = %11lld bytes\n", num_circuit_wires, wire_size, num_circuit_wires * wire_size)
    var total = ((num_circuit_nodes * node_size) + (num_circuit_wires * wire_size))
    c.printf("  Total                            %11lld bytes\n", total)
  end

  var ts1 = c.legion_get_current_time_in_micros()
  load_circuit(__runtime(), __context(),
	       __physical(all_nodes), __fields(all_nodes),
	       __physical(all_wires), __fields(all_wires),
	       conf)

  var ts2 = c.legion_get_current_time_in_micros()
  var colorings =
    create_colorings(__runtime(), __context(),
                 __physical(all_nodes), __fields(all_nodes),
                 __physical(all_wires), __fields(all_wires),
                 conf)

  var ts3 = c.legion_get_current_time_in_micros()
  var rp_all_nodes = partition(disjoint, all_nodes, colorings.privacy_map)
  var all_private = rp_all_nodes[0]
  var all_shared = rp_all_nodes[1]
  var rp_private = partition(disjoint, all_private, colorings.private_node_map)
  var rp_shared = partition(disjoint, all_shared, colorings.shared_node_map)
  var rp_ghost = partition(aliased, all_shared, colorings.ghost_node_map)

  var rp_all_wires = partition(disjoint, all_wires, colorings.wire_owner_map)

  var ts4 = c.legion_get_current_time_in_micros()
  for i = 0, conf.num_pieces do
    var _ = init_pointers(rp_private[i], rp_shared[i], rp_ghost[i], rp_all_wires[i])
    wait_for(_)
  end
  var ts5 = c.legion_get_current_time_in_micros()
  c.printf("partitioning time: load=%llu color=%llu part=%llu check=%llu\n",
	   ts2 - ts1, ts3 - ts2, ts4 - ts3, ts5 - ts4);

  -- Force all previous tasks to complete before continuing.
  do
    var _ = 0
    for i = 0, conf.num_pieces do
      _ += dummy(rp_private[i], rp_shared[i], rp_ghost[i], rp_all_wires[i])
    end
    wait_for(_)
  end

  c.printf("Starting main simulation loop\n")
  var ts_start = c.legion_get_current_time_in_micros()
  var simulation_success = true
  for j = 0, conf.num_loops do
    c.legion_runtime_begin_trace(__runtime(), __context(), 0, false)

    var steps = conf.steps
    if conf.use_dense_kernel then
      var dt : float = DELTAT
      var first_wires : &c.legion_ptr_t = colorings.first_wires
      var num_wires : uint = conf.wires_per_piece
      __demand(__parallel)
      for i = 0, conf.num_pieces do
        dense_calculate_new_currents(steps, dt, first_wires[i], num_wires,
                                     rp_private[i], rp_shared[i], rp_ghost[i],
                                     rp_all_wires[i])
      end
    else
      __demand(__parallel)
      for i = 0, conf.num_pieces do
        calculate_new_currents(steps, rp_private[i], rp_shared[i], rp_ghost[i], rp_all_wires[i])
      end
    end
    __demand(__parallel)
    for i = 0, conf.num_pieces do
      distribute_charge(rp_private[i], rp_shared[i], rp_ghost[i], rp_all_wires[i])
    end
    __demand(__parallel)
    for i = 0, conf.num_pieces do
      update_voltages(rp_private[i], rp_shared[i])
    end

    c.legion_runtime_end_trace(__runtime(), __context(), 0)
  end
  -- Force all previous tasks to complete before continuing.
  do
    var _ = 0
    for i = 0, conf.num_pieces do
      _ += dummy(rp_private[i], rp_shared[i], rp_ghost[i], rp_all_wires[i])
    end
    wait_for(_)
  end
  var ts_end = c.legion_get_current_time_in_micros()
  if simulation_success then
    c.printf("SUCCESS!\n")
  else
    c.printf("FAILURE!\n")
  end
  do
    var sim_time = 1e-6 * (ts_end - ts_start)
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
  if conf.dump_values then
    for i = 0, conf.num_pieces do
      dump_task(rp_private[i], rp_shared[i], rp_ghost[i], rp_all_wires[i])
    end
  end
  c.free(colorings.first_wires)
end
ccircuit.register_mappers()
regentlib.start(toplevel)
