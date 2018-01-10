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

require('legionlib')
require('circuit_defs')
require('circuit_cpu')

function calculate_currents_task(binding, regions,
                                 global_args, local_args, point)
   print(string.format("CPU calculate currents for point %d", point:get_index()))
   calc_new_currents_cpu(regions, global_args, local_args)
end

function distribute_charge_task(binding, regions,
                                global_args, local_args, point)
   print(string.format("CPU distribute charge for point %d", point:get_index()))
   distribute_charge_cpu(regions, global_args, local_args)
end

function update_voltages_task(binding, regions,
                              global_args, local_args, point)
   print(string.format("CPU update voltages for point %d", point:get_index()))
   update_voltages_cpu(regions, global_args, local_args)
end

--- Parse command-line arguments and override configuration accordingly.
-- @params args command-line arguments
-- @params conf simulation configuration (mutable)
function parse_input_args(args, conf)
   for k, v in pairs(args)
   do
      if args[k] == "-l" then
         conf.num_loops = tonumber(args[k + 1])
      end
      if args[k] == "-p" then
         conf.num_pieces = tonumber(args[k + 1])
      end
      if args[k] == "-npp" then
         conf.nodes_per_piece = tonumber(args[k + 1])
      end
      if args[k] == "-wpp" then
         conf.wires_per_piece = tonumber(args[k + 1])
      end
      if args[k] == "-pct" then
         conf.pct_wire_in_piece = tonumber(args[k + 1])
      end
      if args[k] == "-s" then
         conf.random_seed = tonumber(args[k + 1])
      end
   end
end
                 
local function random_element(set)
   return set:get(math.random(set:size() - 1))
end

local function find_location(ptr, private_nodes, shared_nodes, ghost_nodes)
   if private_nodes:mem(ptr) then
      return PRIVATE_PTR
   else
      if shared_nodes:mem(ptr) then
         return SHARED_PTR
      else
         if ghost_nodes:mem(ptr) then
            return GHOST_PTR
         end
      end
   end

   -- Should never make it here, if we do something bad happened
   assert(false)
   return PRIVATE_PTR
end

local function load_circuit(ckt, conf, binding)
   local pieces = {}
   local parts = {}

   print("Initializing circuit simulation...")

   local wires_req = RegionRequirement:new { region = ckt.all_wires } 
   wires_req:add_field(ckt.wire_field)
   local nodes_req = RegionRequirement:new { region = ckt.all_nodes } 
   nodes_req:add_field(ckt.node_field)
   local locator_req = RegionRequirement:new { region = ckt.node_locator } 
   locator_req:add_field(ckt.locator_field)

   local wires = binding:map_region(wires_req)
   local nodes = binding:map_region(nodes_req)
   local locator = binding:map_region(locator_req)

   local wire_owner_map = {}
   local private_node_map = {}
   local shared_node_map = {}
   local ghost_node_map = {}
   local locator_node_map = {}
   local privacy_map = {}
   privacy_map[0] = ColoredPoints:new()
   privacy_map[1] = ColoredPoints:new()
   
   nodes:wait_until_valid()
   local nodes_acc = nodes:get_lua_accessor(ckt.node_field)
   
   local all_node_ispace = ckt.all_nodes.index_space
   binding:allocate_in_indexspace(all_node_ispace,
                                  conf.num_pieces * conf.nodes_per_piece)

   local first_nodes = {}

   local itr = IndexIterator:new(all_node_ispace)
   for n = 0, conf.num_pieces - 1
   do
      private_node_map[n] = ColoredPoints:new()
      locator_node_map[n] = ColoredPoints:new()
      
      for i = 0, conf.nodes_per_piece - 1
      do
         assert(itr:has_next())
         local node_ptr = itr:next()
         if i == 0 then first_nodes[n] = node_ptr end
         local node = {}
         node.charge = 0.0
         node.voltage = 2 * math.random() - 1
         node.capacitance = math.random() + 1
         node.leakage = 0.1 * math.random()
         nodes_acc:write(node_ptr, node)

         -- Just put everything in everyones private map at the moment       
         -- We'll pull pointers out of here later as nodes get tied to 
         -- wires that are non-local
         private_node_map[n]:add(node_ptr)
         privacy_map[0]:add(node_ptr)
         locator_node_map[n]:add(node_ptr)
      end
   end

   print("Done")

   wires:wait_until_valid()
   local wires_acc = wires:get_lua_accessor(ckt.wire_field)
   
   -- Allocate all the wires
   local all_wire_ispace = ckt.all_wires.index_space
   binding:allocate_in_indexspace(all_wire_ispace,
                                  conf.num_pieces * conf.wires_per_piece)

   local first_wires = {}

   itr = IndexIterator:new(all_wire_ispace)
   for n = 0, conf.num_pieces - 1
   do
      wire_owner_map[n] = ColoredPoints:new()
      ghost_node_map[n] = ColoredPoints:new()
      
      for i = 0, conf.wires_per_piece - 1
      do
         assert(itr:has_next())
         local wire_ptr = itr:next()
         -- Record the first wire pointer for this piece
         if i == 0 then first_wires[i] = wire_ptr end
         local wire = {}
         wire.current = {}
         for j = 0, WIRE_SEGMENTS - 1 do wire.current[j] = 0.0 end
         wire.voltage = {}
         for j = 0, WIRE_SEGMENTS - 2 do wire.voltage[j] = 0.0 end
         wire.resistance = math.random() * 10 + 1
         wire.inductance = math.random() * 0.01 + 0.1
         wire.capacitance = math.random() * 0.1

         wire.in_ptr = random_element(private_node_map[n])

         if math.random(0, 100) < conf.pct_wire_in_piece
         then
            wire.out_ptr = random_element(private_node_map[n])
         else
            -- pick a random other piece and a node from there
            local nn = n

            while (n == nn) do nn = math.random(conf.num_pieces) - 1 end

            wire.out_ptr = random_element(private_node_map[nn])

           -- This node is no longer private
            privacy_map[0]:del(wire.out_ptr)
            privacy_map[1]:add(wire.out_ptr)
            ghost_node_map[n]:add(wire.out_ptr)
         end

         -- Write the wire
         wires_acc:write(wire_ptr, wire)
         wire_owner_map[n]:add(wire_ptr)
      end
   end

   print("Done")

   locator:wait_until_valid()
   local locator_acc = locator:get_lua_accessor(ckt.locator_field)

   -- Second pass: make some random fraction of the private nodes shared

   itr = IndexIterator:new(all_node_ispace)
   for n = 0, conf.num_pieces - 1
   do
      shared_node_map[n] = ColoredPoints:new()
      for i = 0, conf.nodes_per_piece - 1
      do
         assert(itr:has_next())
         local node_ptr = itr:next()
         if not privacy_map[0]:mem(node_ptr)
         then
            private_node_map[n]:del(node_ptr)
            -- node is now shared
            shared_node_map[n]:add(node_ptr)
            locator_acc:write(node_ptr, { loc = SHARED_PTR })
         else
            -- node is private 
            locator_acc:write(node_ptr, { loc = PRIVATE_PTR })
         end
      end
   end

   print("Done")

   -- Second pass (part 2): go through the wires and update the locations

   itr = IndexIterator:new(all_wire_ispace)
   for n = 0, conf.num_pieces - 1
   do
      for i = 0, conf.wires_per_piece - 1
      do
         assert(itr:has_next())
         local wire_ptr = itr:next()
         local wire = wires_acc:read(wire_ptr)

         wire.in_loc =
            find_location(wire.in_ptr,
                          private_node_map[n],
                          shared_node_map[n],
                          ghost_node_map[n])
         wire.out_loc = 
            find_location(wire.out_ptr,
                          private_node_map[n],
                          shared_node_map[n],
                          ghost_node_map[n])

         -- Write the wire back
         wires_acc:write(wire_ptr, wire)
      end
   end

   print("Done")

   binding:unmap_region(wires)
   binding:unmap_region(nodes)
   binding:unmap_region(locator)
         
   -- Now we can create our partitions and update the circuit pieces

   local privacy_part =
      binding:create_index_partition(all_node_ispace, privacy_map, true)
   local all_private = binding:get_index_subspace(privacy_part, 0)
   local all_shared  = binding:get_index_subspace(privacy_part, 1)

   -- Now create partitions for each of the subregions
   local priv =
      binding:create_index_partition(all_private,
                                     private_node_map, true)
   parts.pvt_nodes =
      binding:get_logical_partition_by_tree(priv,
                                            ckt.all_nodes.field_space,
                                            ckt.all_nodes.tree_id)
   local shared =
      binding:create_index_partition(all_shared,
                                     shared_node_map, true)
   parts.shr_nodes =
      binding:get_logical_partition_by_tree(shared,
                                            ckt.all_nodes.field_space,
                                            ckt.all_nodes.tree_id)
   local ghost =
      binding:create_index_partition(all_shared,
                                     ghost_node_map, false)
   parts.ghost_nodes =
      binding:get_logical_partition_by_tree(ghost,
                                            ckt.all_nodes.field_space,
                                            ckt.all_nodes.tree_id)
   local pvt_wires =
      binding:create_index_partition(all_wire_ispace,
                                     wire_owner_map, true)
   parts.pvt_wires =
      binding:get_logical_partition_by_tree(pvt_wires,
                                            ckt.all_wires.field_space,
                                            ckt.all_wires.tree_id); 
   local locs =
      binding:create_index_partition(ckt.node_locator.index_space,
                                     locator_node_map, true)
   parts.node_locations =
      binding:get_logical_partition_by_tree(locs,
                                            ckt.node_locator.field_space,
                                            ckt.node_locator.tree_id)

   -- Build the pieces
   for n = 0, conf.num_pieces - 1
   do
      pieces[n] = {}
      pieces[n].pvt_nodes =
         binding:get_logical_subregion_by_color(parts.pvt_nodes, n)
      pieces[n].shr_nodes =
         binding:get_logical_subregion_by_color(parts.shr_nodes, n)
      pieces[n].ghost_nodes =
         binding:get_logical_subregion_by_color(parts.ghost_nodes, n)
      pieces[n].pvt_wires =
         binding:get_logical_subregion_by_color(parts.pvt_wires, n)
      pieces[n].num_wires = conf.wires_per_piece
      pieces[n].first_wire = first_wires[n]
      pieces[n].num_nodes = conf.nodes_per_piece
      pieces[n].first_node = first_nodes[n]
   end
   
   print("Done")

   return pieces, parts
end

function circuit_main(binding, regions, args)
   local conf = {
      num_loops = 2,
      num_pieces = 4,
      nodes_per_piece = 2,
      wires_per_piece = 4,
      pct_wire_in_piece = 95,
      random_seed = 12345
   }

   parse_input_args(args, conf)

   print(string.format("circuit settings: loops=%d pieces=%d " ..
                          "nodes/piece=%d wires/piece=%d " ..
                          "pct_in_piece=%d seed=%d",
                       conf.num_loops, conf.num_pieces,
                       conf.nodes_per_piece, conf.wires_per_piece,
                       conf.pct_wire_in_piece, conf.random_seed))
   
   math.randomseed(conf.random_seed)

   local circuit = {}

   local num_circuit_nodes = conf.num_pieces * conf.nodes_per_piece
   local num_circuit_wires = conf.num_pieces * conf.wires_per_piece
   -- Make index spaces
   local node_index_space = binding:create_index_space(num_circuit_nodes)
   local wire_index_space = binding:create_index_space(num_circuit_wires)
   -- Make field spaces
   local node_field_space = binding:create_field_space()
   local wire_field_space = binding:create_field_space()
   local locator_field_space = binding:create_field_space()

   -- Allocate fields
   circuit.node_field =
      binding:allocate_field(node_field_space, CircuitNode)
   circuit.wire_field =
      binding:allocate_field(wire_field_space, CircuitWire)
   circuit.locator_field =
      binding:allocate_field(locator_field_space, PointerLocation)

   -- Make logical regions
   circuit.all_nodes =
      binding:create_logical_region(node_index_space, node_field_space)
   circuit.all_wires =
      binding:create_logical_region(wire_index_space, wire_field_space)
   circuit.node_locator =
      binding:create_logical_region(node_index_space, locator_field_space)

   -- Load the circuit
   local pieces, parts = load_circuit(circuit, conf, binding)

   -- Start the simulation
   print("Starting main simulation loop")

   -- Prepare arguments
   local global_arg =
      { node_field = circuit.node_field,
        wire_field = circuit.wire_field }
   local local_args = {}
   for idx = 0, conf.num_pieces - 1
   do
      local_args[idx] = pieces[idx]
   end

   local task_space = binding:create_index_space(conf.num_pieces)
   binding:allocate_in_indexspace(task_space, conf.num_pieces)
   local domain = Domain:new(task_space)

   local cnc_launcher = IndexLauncher:new(CALC_NEW_CURRENTS, domain,
                                          global_arg, local_args)
   local dsc_launcher = IndexLauncher:new(DISTRIBUTE_CHARGE, domain,
                                          global_arg, local_args)
   local upv_launcher = IndexLauncher:new(UPDATE_VOLTAGES, domain,
                                          global_arg, local_args)

   -- Build the region requirements for each task
   cnc_launcher:add_region_requirements(
      RegionRequirement:new { part = parts.pvt_wires,
                              parent = circuit.all_wires }
         :add_field(circuit.wire_field),

      RegionRequirement:new { part = parts.pvt_nodes,
                              priv = PrivilegeMode.READ_ONLY,
                              parent = circuit.all_nodes }
         :add_field(circuit.node_field),

      RegionRequirement:new { part = parts.shr_nodes,
                              priv = PrivilegeMode.READ_ONLY,
                              parent = circuit.all_nodes }
         :add_field(circuit.node_field),

      RegionRequirement:new { part = parts.ghost_nodes,
                              priv = PrivilegeMode.READ_ONLY,
                              parent = circuit.all_nodes }
         :add_field(circuit.node_field)
   )
      
   dsc_launcher:add_region_requirements(
      RegionRequirement:new { part = parts.pvt_wires,
                              priv = PrivilegeMode.READ_ONLY,
                              parent = circuit.all_wires}
         :add_field(circuit.wire_field),

      RegionRequirement:new { part = parts.pvt_nodes,
                              parent = circuit.all_nodes }
         :add_field(circuit.node_field),

      RegionRequirement:new { part = parts.shr_nodes,
                              reduce_op = REDUCE_ID,
                              parent = circuit.all_nodes }
         :add_field(circuit.node_field),

      RegionRequirement:new { part = parts.ghost_nodes,
                              reduce_op = REDUCE_ID,
                              parent = circuit.all_nodes }
         :add_field(circuit.node_field)
   )

   upv_launcher:add_region_requirements(
      RegionRequirement:new { part = parts.pvt_nodes,
                              parent = circuit.all_nodes }
         :add_field(circuit.node_field),

      RegionRequirement:new { part = parts.shr_nodes,
                              parent = circuit.all_nodes }
         :add_field(circuit.node_field),

      RegionRequirement:new { part = parts.node_locations,
                              priv = PrivilegeMode.READ_ONLY,
                              parent = circuit.node_locator }
         :add_field(circuit.locator_field)
   )
   
   local ts_start = os.time()
   local last = nil
   for i = 1, conf.num_loops
   do
      print(string.format("starting loop %d of %d", i, conf.num_loops))

      binding:execute_index_space(cnc_launcher)
      binding:execute_index_space(dsc_launcher)
      last = binding:execute_index_space(upv_launcher)
   end

   print("waiting for all simulation tasks to complete")
   last:wait_all_results()
   local ts_end = os.time()

   print("simulation complete - destroying regions")

   local sim_time = ts_end - ts_start
   local num_circuit_nodes = conf.num_pieces * conf.nodes_per_piece
   local num_circuit_wires = conf.num_pieces * conf.wires_per_piece
   local operations = num_circuit_wires * (WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * STEPS
   operations = operations + (num_circuit_wires * 4)
   operations = operations + (num_circuit_nodes * 4)
   operations = operations * conf.num_loops
   local gflops = (1e-9 * operations) / sim_time
   print("GFLOPS = " .. gflops .. " GFLOPS")


   -- Now we can destroy all the things that we made

   binding:destroy_task_arguments()
   binding:destroy_logical_region(circuit.all_nodes)
   binding:destroy_logical_region(circuit.all_wires)
   binding:destroy_logical_region(circuit.node_locator)
   binding:destroy_field_space(node_field_space)
   binding:destroy_field_space(wire_field_space)
   binding:destroy_field_space(locator_field_space)
   binding:destroy_index_space(node_index_space)
   binding:destroy_index_space(wire_index_space)
   binding:destroy_index_space(task_space)
end
