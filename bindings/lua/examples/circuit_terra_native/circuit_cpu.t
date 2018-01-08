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
terralib.require('circuit_defs')

local node_accessor_type = TerraAccessor(CircuitNode)
local wire_accessor_type = TerraAccessor(CircuitWire)
local reducer_type = TerraReducer(CircuitNode, "charge")

local std = terralib.includec("stdio.h")

local terra get_node(priv: node_accessor_type,
               shr: node_accessor_type,
               ghost: node_accessor_type,
               loc: int, ptr: uint32)
   if loc == PRIVATE_PTR
   then return priv:read(ptr)
   else if loc == SHARED_PTR
        then return shr:read(ptr)
        else if loc == GHOST_PTR
             then return ghost:read(ptr)
             end
        end
   end
end   
-- get_node:compile()

function gen_calc_new_currents_cpu(WS, NS)
   local ptype = &float
   local vec  = vector(float, 4)
   local vecp = &vec

   local function alignedload(addr)
      return `terralib.attrload(addr, { align = 4 })
   end
   local function alignedstore(addr, v)
      return `terralib.attrstore(addr, v, { align = 4 })
   end
   alignedload, alignedstore = macro(alignedload),macro(alignedstore)

   return
      terra(regions: &TPhysicalRegion, conf: &Circuit, piece: &CircuitPiece)
         var pvt_wires: wire_accessor_type
         var pvt_nodes: node_accessor_type
         var shr_nodes: node_accessor_type
         var ghost_nodes: node_accessor_type

         pvt_wires:init_with_field(regions[0], conf.wire_field)
         pvt_nodes:init_with_field(regions[1], conf.node_field)
         shr_nodes:init_with_field(regions[2], conf.node_field)
         ghost_nodes:init_with_field(regions[3], conf.node_field)

         var itr:TerraIndexIterator
         itr:init(piece.pvt_wires)

         while itr:has_next()
         do
            var wire_ptr = itr:next()

            var wire = pvt_wires:read(wire_ptr)
            var in_node  = get_node(pvt_nodes, shr_nodes, ghost_nodes,
                                    wire.in_loc, wire.in_ptr)
            var out_node = get_node(pvt_nodes, shr_nodes, ghost_nodes,
                                    wire.out_loc, wire.out_ptr)

            -- Solve RLC model iteratively
            var new_i : float[10]
            var new_i_ptr: ptype = [ptype](new_i)
            var new_v : float[11]
            var new_v_ptr: ptype = [ptype](new_v)

            var wire_current : ptype = [ptype](wire.current) 
            var wire_voltage : ptype = [ptype](wire.voltage)

            alignedstore(vecp(&new_i_ptr[0]), alignedload(vecp(&wire_current[0])))
            alignedstore(vecp(&new_i_ptr[4]), alignedload(vecp(&wire_current[4])))
            new_i_ptr[8] = wire_current[8]
            new_i_ptr[9] = wire_current[9]
            new_v_ptr[0] = in_node.voltage
            alignedstore(vecp(&new_v_ptr[1]), alignedload(vecp(&wire_voltage[0])))
            alignedstore(vecp(&new_v_ptr[5]), alignedload(vecp(&wire_voltage[4])))
            new_v_ptr[9] = wire_voltage[8];
            new_v_ptr[WS] = out_node.voltage

            var dt_vec : vec  = DELTAT
            var cap_vec : vec = wire.capacitance
            var ind_vec : vec = wire.inductance
            var res_vec : vec = wire.resistance

            var dt : float = DELTAT
            var cap : float = wire.capacitance
            var ind : float = wire.inductance
            var res : float = wire.resistance

            for j = 0, NS
            do
               for i = 0, 8, 4
               do
                  var tmp1 : vec = alignedload(vecp(&new_v_ptr[i + 1])) - alignedload(vecp(&new_v_ptr[i]))
                  var tmp2 : vec = alignedload(vecp(&new_i_ptr[i])) - alignedload(vecp(&wire_current[i]))
                  tmp2 = ind_vec * tmp2 / dt_vec
                  alignedstore(vecp(&new_i_ptr[i]), (tmp2 - tmp1) / res_vec)
               end

               for i = 8, 10
               do
                  var tmp1 = new_v_ptr[i] - new_v_ptr[i + 1]
                  var tmp2 = new_i_ptr[i] - wire_current[i]
                  tmp2 = ind * tmp2 / dt
                  new_i_ptr[i] = (tmp1 - tmp2) / res
               end

               for i = 0, 3
               do
                  var tmp1 = wire_voltage[i]
                  var tmp2 = new_i_ptr[i] - new_i_ptr[i + 1] 
                  new_v_ptr[i + 1] = tmp1 + dt * tmp2 / cap
               end

               alignedstore(vecp(&new_v_ptr[4]),
                  alignedload(vecp(&wire_voltage[3])) +
                  dt_vec * (alignedload(vecp(&new_i_ptr[3])) - 
                            alignedload(vecp(&new_i_ptr[4])) / cap_vec))

               for i = 7, 9
               do
                  var tmp1 = wire_voltage[i]
                  var tmp2 = new_i_ptr[i] - new_i_ptr[i + 1] 
                  new_v_ptr[i + 1] = tmp1 + dt * tmp2 / cap
               end

            end

            alignedstore(vecp(&wire_current[0]), alignedload(vecp(&new_i_ptr[0])))
            alignedstore(vecp(&wire_current[4]), alignedload(vecp(&new_i_ptr[4])))
            wire_current[8] = new_i_ptr[8]
            wire_current[9] = new_i_ptr[9]
            alignedstore(vecp(&wire_voltage[0]), alignedload(vecp(&new_v_ptr[1])))
            alignedstore(vecp(&wire_voltage[4]), alignedload(vecp(&new_v_ptr[5])))
            wire_voltage[8] = new_v_ptr[9]

            pvt_wires:write(wire_ptr, wire)
         end

         itr:close()

         pvt_wires:close()
         pvt_nodes:close()
         shr_nodes:close()
         ghost_nodes:close()
      end
end

calc_new_currents_cpu_vectorized = gen_calc_new_currents_cpu(WIRE_SEGMENTS, STEPS)
-- calc_new_currents_cpu:compile()

terra calc_new_currents_cpu(regions: &TPhysicalRegion,
                            conf: &Circuit,
                            piece: &CircuitPiece)

   var pvt_wires: wire_accessor_type
   var pvt_nodes: node_accessor_type
   var shr_nodes: node_accessor_type
   var ghost_nodes: node_accessor_type

   pvt_wires:init_with_field(regions[0], conf.wire_field)
   pvt_nodes:init_with_field(regions[1], conf.node_field)
   shr_nodes:init_with_field(regions[2], conf.node_field)
   ghost_nodes:init_with_field(regions[3], conf.node_field)

   var itr:TerraIndexIterator
   itr:init(piece.pvt_wires)

   while itr:has_next()
   do
      var wire_ptr = itr:next()

      var wire = pvt_wires:read(wire_ptr)
      var in_node  = get_node(pvt_nodes, shr_nodes, ghost_nodes,
                              wire.in_loc, wire.in_ptr)
      var out_node = get_node(pvt_nodes, shr_nodes, ghost_nodes,
                              wire.out_loc, wire.out_ptr)

      -- Solve RLC model iteratively
      var new_i : float[WIRE_SEGMENTS]
      var new_v : float[WIRE_SEGMENTS + 1]

      var dt = DELTAT
      var steps = STEPS

      for i = 0, WIRE_SEGMENTS     do new_i[i] = wire.current[i] end
      for i = 0, WIRE_SEGMENTS - 1 do new_v[i + 1] = wire.voltage[i] end

      new_v[0] = in_node.voltage
      new_v[WIRE_SEGMENTS] = out_node.voltage

      for j = 0, steps
      do
         -- first, figure out the new current from the voltage differential
         -- and our inductance:
         -- dV = R*I + L*I' ==> I = (dV - L*I')/R
         for i = 0, WIRE_SEGMENTS
         do
            new_i[i] =
               ((new_v[i] - new_v[i + 1]) - 
                   (wire.inductance * (new_i[i] - wire.current[i]) / dt))
               / wire.resistance
         end
         -- Now update the inter-node voltages
         for i = 0, WIRE_SEGMENTS - 1
         do
            new_v[i + 1] =
               wire.voltage[i] +
               dt * (new_i[i] - new_i[i + 1]) / wire.capacitance
         end
      end

      for i = 0, WIRE_SEGMENTS     do wire.current[i] = new_i[i] end
      for i = 0, WIRE_SEGMENTS - 1 do wire.voltage[i] = new_v[i + 1] end

      pvt_wires:write(wire_ptr, wire)
   end

   itr:close()

   pvt_wires:close()
   pvt_nodes:close()
   shr_nodes:close()
   ghost_nodes:close()
end

terra calculate_currents_task_terra(
      conf: &Circuit, piece: &CircuitPiece, task: TTask,
      regions: &TPhysicalRegion, num_regions: uint64,
      ctx: &opaque, runtime: &opaque)
   std.printf("CPU calculate currents for point %d\n", task:get_index())
   if conf.vectorize == 1 then 
      calc_new_currents_cpu_vectorized(regions, conf, piece)
   else
      calc_new_currents_cpu(regions, conf, piece)
   end
end
-- calculate_currents_task_terra:compile()

terra reduce_node(pvt: reducer_type,
                  shr: reducer_type,
                  ghost: reducer_type,
                  loc: int, ptr: int, value: float)
   if loc == PRIVATE_PTR
   then
      pvt:reduce(ptr, value)
   else if loc == SHARED_PTR
        then
           shr:reduce(ptr, value)
        else if loc == GHOST_PTR
             then
                ghost:reduce(ptr, value)
             end
        end
   end
end

terra distribute_charges_cpu(regions: &TPhysicalRegion,
                             piece: &CircuitPiece)
   var pvt_wires: wire_accessor_type
   var pvt_nodes: reducer_type
   var shr_nodes: reducer_type
   var ghost_nodes: reducer_type

   pvt_wires:init(regions[0])
   pvt_nodes:init(regions[1], ReductionType.PLUS)
   shr_nodes:init(regions[2], ReductionType.PLUS)
   ghost_nodes:init(regions[3], ReductionType.PLUS)

   var itr: TerraIndexIterator
   itr:init(piece.pvt_wires)
   while itr:has_next()
   do
      var wire_ptr = itr:next()
      var wire = pvt_wires:read(wire_ptr)

      var dt : float = DELTAT; 

      reduce_node(pvt_nodes, shr_nodes, ghost_nodes,
                  wire.in_loc, wire.in_ptr,
                     -dt * wire.current[0])
      reduce_node(pvt_nodes, shr_nodes, ghost_nodes,
                  wire.out_loc, wire.out_ptr,
                     dt * wire.current[WIRE_SEGMENTS - 1])
   end

   pvt_wires:close()
   pvt_nodes:close()
   shr_nodes:close()
   ghost_nodes:close()
end

terra distribute_charges_task_terra(
      conf: &Circuit, piece: &CircuitPiece, task: TTask,
      regions: &TPhysicalRegion, num_regions: uint64,
      ctx: &opaque, runtime: &opaque)
   std.printf("CPU distribute charge for point %d\n", task:get_index())
   distribute_charges_cpu(regions, piece)
end

terra update_region_voltages(piece: &CircuitPiece,
                             nodes: node_accessor_type,
                             itr: TerraIndexIterator)
   while itr:has_next()
   do
      var node_ptr = itr:next()
      var node: CircuitNode = nodes:read(node_ptr)

      -- charge adds in, and then some leaks away
      node.voltage = node.voltage + node.charge / node.capacitance
      node.voltage = node.voltage * (1.0 - node.leakage)
      node.charge = 0

      nodes:write(node_ptr, node)
   end
end
-- update_region_voltages:compile()

terra update_voltages_cpu(regions: &TPhysicalRegion,
                          conf: &Circuit, piece: &CircuitPiece)
   
   var pvt_nodes: node_accessor_type
   var shr_nodes: node_accessor_type

   pvt_nodes:init_with_field(regions[0], conf.node_field)
   shr_nodes:init_with_field(regions[1], conf.node_field)

   var itr: TerraIndexIterator
   itr:init(piece.pvt_nodes)
   update_region_voltages(piece, pvt_nodes, itr)
   itr:close()

   itr:init(piece.shr_nodes)
   update_region_voltages(piece, shr_nodes, itr)
   itr:close()

   pvt_nodes:close()
   shr_nodes:close()
end
-- update_voltages_cpu:compile()

terra update_voltages_task_terra(
      conf: &Circuit, piece: &CircuitPiece, task: TTask,
      regions: &TPhysicalRegion, num_regions: uint64,
      ctx: &opaque, runtime: &opaque)
   std.printf("CPU update voltages for point %d\n", task:get_index())
   update_voltages_cpu(regions, conf, piece)
end
-- update_voltages_task_terra:compile()
