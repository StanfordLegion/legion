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

local function get_node(priv, shr, ghost, loc, ptr)
   if loc == PRIVATE_PTR
   then return priv:read(ptr)
   else if loc == SHARED_PTR
        then return shr:read(ptr)
        else if loc == GHOST_PTR
             then return ghost:read(ptr)
             else assert(false)
             end
        end
   end
end   
      

function calc_new_currents_cpu(reg, global_args, piece)
   local pvt_wires = reg[0]:get_lua_accessor(global_args.wire_field)
   local pvt_nodes = reg[1]:get_lua_accessor(global_args.node_field)
   local shr_nodes = reg[2]:get_lua_accessor(global_args.node_field)
   local ghost_nodes = reg[3]:get_lua_accessor(global_args.node_field)

   local itr = IndexIterator:new(piece.pvt_wires)

   while itr:has_next()
   do
      local wire_ptr = itr:next()

      local wire = pvt_wires:read(wire_ptr)
      local in_node  = get_node(pvt_nodes, shr_nodes, ghost_nodes,
                                wire.in_loc, wire.in_ptr)
      local out_node = get_node(pvt_nodes, shr_nodes, ghost_nodes,
                                wire.out_loc, wire.out_ptr)

      -- Solve RLC model iteratively
      local dt = DELTAT
      local steps = STEPS
      local new_v = {}
      local new_i = {}
      for i = 0, WIRE_SEGMENTS - 1 do new_i[i] = wire.current[i] end
      for i = 0, WIRE_SEGMENTS - 2 do new_v[i + 1] = wire.voltage[i] end
      new_v[0] = in_node.voltage
      new_v[WIRE_SEGMENTS] = out_node.voltage

      for j = 1, steps
      do
         -- first, figure out the new current from the voltage differential
         -- and our inductance:
         -- dV = R*I + L*I' ==> I = (dV - L*I')/R
         for i = 0, WIRE_SEGMENTS - 1
         do
            new_i[i] =
               ((new_v[i + 1] - new_v[i]) - 
                   (wire.inductance * (new_i[i] - wire.current[i]) / dt))
               / wire.resistance
            --print(string.format("[step %d] new_i[%d] = %f", j, i, new_i[i]))
         end
         -- Now update the inter-node voltages
         for i = 0, WIRE_SEGMENTS - 2
         do
            new_v[i + 1] =
               wire.voltage[i] +
               dt * (new_i[i] - new_i[i + 1]) / wire.capacitance
            --print(string.format("[step %d] new_v[%d] = %f", j, i + 1, new_v[i + 1]))
         end
      end

      -- Copy everything back
      for i = 0, WIRE_SEGMENTS - 1 do wire.current[i] = new_i[i] end
      for i = 0, WIRE_SEGMENTS - 2 do wire.voltage[i] = new_v[i + 1] end

      pvt_wires:write(wire_ptr, wire)
   end
end

local function reduce_node(pvt, shr, ghost, loc, ptr, value)
   if loc == PRIVATE_PTR
   then
      pvt.charge:reduce(ReductionType.PLUS, ptr, value)
   else if loc == SHARED_PTR
        then
           shr.charge:reduce(ReductionType.PLUS, ptr, value)
        else if loc == GHOST_PTR
             then
                ghost.charge:reduce(ReductionType.PLUS, ptr, value)
             else
                assert(false)
             end
        end
   end
end

function distribute_charge_cpu(reg, global_args, piece)
   local pvt_wires = reg[0]:get_lua_accessor(global_args.wire_field)
   local pvt_nodes = reg[1]:get_lua_accessor(global_args.node_field)
   local shr_nodes = reg[2]:get_lua_accessor(global_args.node_field)
   local ghost_nodes = reg[3]:get_lua_accessor(global_args.node_field)

   local itr = IndexIterator:new(piece.pvt_wires)
   while itr:has_next()
   do
      local wire_ptr = itr:next()
      local wire = pvt_wires:read(wire_ptr)

      local dt = DELTAT; 

      reduce_node(pvt_nodes, shr_nodes, ghost_nodes,
                  wire.in_loc, wire.in_ptr,
                     -dt * wire.current[0])
      reduce_node(pvt_nodes, shr_nodes, ghost_nodes,
                  wire.out_loc, wire.out_ptr,
                     dt * wire.current[WIRE_SEGMENTS - 1])
   end

end

local function update_region_voltages(piece, nodes, itr)
   while itr:has_next()
   do
      local node_ptr = itr:next()
      local node = nodes:read(node_ptr)

      -- charge adds in, and then some leaks away
      node.voltage = node.voltage + node.charge / node.capacitance
      node.voltage = node.voltage * (1.0 - node.leakage)
      node.charge = 0

      nodes:write(node_ptr, node)
   end
end

function update_voltages_cpu(reg, global_args, piece)
   local pvt_nodes = reg[0]:get_lua_accessor(global_args.node_field)
   local shr_nodes = reg[1]:get_lua_accessor(global_args.node_field)

   local itr = IndexIterator:new(piece.pvt_nodes)
   update_region_voltages(piece, pvt_nodes, itr)
   itr = IndexIterator:new(piece.shr_nodes)
   update_region_voltages(piece, shr_nodes, itr)
end
