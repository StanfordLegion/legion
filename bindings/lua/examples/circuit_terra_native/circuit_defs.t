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

CIRCUIT_MAIN = 100
CALC_NEW_CURRENTS = 200
DISTRIBUTE_CHARGE = 300
UPDATE_VOLTAGES = 400

PRIVATE_PTR = 0
SHARED_PTR = 1
GHOST_PTR = 2

WIRE_SEGMENTS = 10
STEPS = 10000
DELTAT = 1e-6

REDUCE_ID = 1

struct CircuitNode
{
   charge : float;
   voltage : float;
   capacitance : float;
   leakage : float;
}

struct CircuitWire
{
   in_ptr : int;
   out_ptr : int;
   in_loc : int;
   out_loc : int;
   inductance : float;
   resistance : float;
   capacitance : float;
   current : float[WIRE_SEGMENTS];
   voltage : float[WIRE_SEGMENTS - 1];
   padd : float[6];
}

struct PointerLocation
{
   loc : int
}

struct Circuit
{
   node_field: int,
   wire_field: int,
   vectorize: int
}

struct CircuitPiece
{
   pvt_nodes: TLogicalRegion,
   shr_nodes: TLogicalRegion,
   ghost_nodes: TLogicalRegion,
   pvt_wires: TLogicalRegion,
   num_wires: int,
   first_wire: int,
   num_nodes: int,
   first_node: int
}

