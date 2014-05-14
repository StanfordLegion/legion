/* Copyright 2013 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __CIRCUIT_H__
#define __CIRCUIT_H__

#include "legion.h"

//#define DISABLE_MATH

#define WIRE_SEGMENTS 10
#define STEPS         10000
#define DELTAT        1e-6

#define INDEX_TYPE    unsigned
#define INDEX_DIM     1

using namespace LegionRuntime::HighLevel;

// Data type definitions

enum PointerLocation {
  PRIVATE_PTR,
  SHARED_PTR,
  GHOST_PTR,
};

enum {
  REGION_MAIN,
  CALC_NEW_CURRENTS,
  DISTRIBUTE_CHARGE,
  UPDATE_VOLTAGES,
};

enum {
  REDUCE_ID = 1,
};

struct CircuitNode {
  float charge;
  float voltage;
  float capacitance;
  float leakage;
};

struct CircuitWire {
  ptr_t in_ptr, out_ptr;
  PointerLocation in_loc, out_loc;
  float inductance;
  float resistance;
  float current[WIRE_SEGMENTS];
  float capacitance;
  float voltage[WIRE_SEGMENTS-1];
  float crap, morecrap, evenmorecrap, yetmorecrap, extracrap, lastcrap;
};

struct Circuit {
  LogicalRegion all_nodes;
  LogicalRegion all_wires;
  LogicalRegion node_locator;
  FieldID node_field;
  FieldID wire_field;
  FieldID locator_field;
  IndexSpace launch_space;
};

struct CircuitPiece {
  LogicalRegion pvt_nodes, shr_nodes, ghost_nodes;
  LogicalRegion pvt_wires;
  unsigned      num_wires;
  ptr_t first_wire;
  unsigned      num_nodes;
  ptr_t first_node;
  float dt;
  int steps;
};

struct Partitions {
  LogicalPartition pvt_wires;
  LogicalPartition pvt_nodes, shr_nodes, ghost_nodes;
  LogicalPartition node_locations;
};

// Reduction Op
class AccumulateCharge {
public:
  typedef CircuitNode LHS;
  typedef float RHS;
  static const float identity;

  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);

  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

// CPU function variants

void calc_new_currents_cpu(CircuitPiece *p,
                           const std::vector<PhysicalRegion> &physical_regions);

void distribute_charge_cpu(CircuitPiece *p,
                           const std::vector<PhysicalRegion> &physical_regions);

void update_voltages_cpu(CircuitPiece *p,
                         const std::vector<PhysicalRegion> &physical_regions);

// Functions for linking against CUDA

void calc_new_currents_gpu(CircuitPiece *p,
                           const std::vector<PhysicalRegion> &physical_regions);

void distribute_charge_gpu(CircuitPiece *p,
                           const std::vector<PhysicalRegion> &physical_regions);

void update_voltages_gpu(CircuitPiece *p,
                         const std::vector<PhysicalRegion> &physical_regions);

#endif // __CIRCUIT_H__

