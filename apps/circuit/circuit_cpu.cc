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


#include "circuit.h"

using namespace LegionRuntime::Accessor;

const float AccumulateCharge::identity = 0.0f;

template <>
void AccumulateCharge::apply<true>(LHS &lhs, RHS rhs) 
{
  lhs.charge += rhs;
}

template <>
void AccumulateCharge::apply<false>(LHS &lhs, RHS rhs) 
{
  // most cpus don't let you atomic add a float, so we use gcc's builtin
  // compare-and-swap in a loop
  int *target = (int *)&(lhs.charge);
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs;
  } while(!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

template <>
void AccumulateCharge::fold<true>(RHS &rhs1, RHS rhs2) 
{
  rhs1 += rhs2;
}

template <>
void AccumulateCharge::fold<false>(RHS &rhs1, RHS rhs2) 
{
  // most cpus don't let you atomic add a float, so we use gcc's builtin
  // compare-and-swap in a loop
  int *target = (int *)&rhs1;
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs2;
  } while(!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

// Helper methods

template<typename AT>
static inline CircuitNode get_node(const RegionAccessor<AT,CircuitNode> &priv,
                                   const RegionAccessor<AT,CircuitNode> &shr,
                                   const RegionAccessor<AT,CircuitNode> &ghost,
                                   PointerLocation loc, ptr_t ptr) 
{
  switch (loc)
  {
    case PRIVATE_PTR:
      return priv.read(ptr);
    case SHARED_PTR:
      return shr.read(ptr);
    case GHOST_PTR:
      return ghost.read(ptr);
    default:
      assert(false);
  }
  return CircuitNode();
}

template<typename REDOP, typename AT1, typename AT2>
static inline void reduce_node(const RegionAccessor<AT1,CircuitNode> &priv,
                               const RegionAccessor<AT2,CircuitNode> &shr,
                               const RegionAccessor<AT2,CircuitNode> &ghost,
                               PointerLocation loc, ptr_t ptr, typename REDOP::RHS value)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      priv.template reduce<REDOP>(ptr, value);
      break;
    case SHARED_PTR:
      shr.template reduce(ptr, value);
      break;
    case GHOST_PTR:
      ghost.template reduce(ptr, value);
      break;
    default:
      assert(false);
  }
}

template<typename AT>
static inline void update_region_voltages(CircuitPiece *p, 
                                          const RegionAccessor<AT,CircuitNode> &nodes,
                                          IndexIterator &itr)
{
  while (itr.has_next())
  {
    ptr_t node_ptr = itr.next();
    CircuitNode node = nodes.read(node_ptr);

    // charge adds in, and then some leaks away
    node.voltage += node.charge / node.capacitance;
    node.voltage *= (1.f - node.leakage);
    node.charge = 0.f;

    nodes.write(node_ptr, node);
  }
}

// Actual implementations

void calc_new_currents_cpu(CircuitPiece *p,
                           const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::Generic, CircuitWire> pvt_wires = regions[0].get_accessor().typeify<CircuitWire>();
  RegionAccessor<AccessorType::Generic, CircuitNode> pvt_nodes = regions[1].get_accessor().typeify<CircuitNode>();
  RegionAccessor<AccessorType::Generic, CircuitNode> shr_nodes = regions[2].get_accessor().typeify<CircuitNode>();
  RegionAccessor<AccessorType::Generic, CircuitNode> ghost_nodes = regions[3].get_accessor().typeify<CircuitNode>();
  LegionRuntime::HighLevel::IndexIterator itr(p->pvt_wires);
  while (itr.has_next())
  {
    ptr_t wire_ptr = itr.next();
    CircuitWire wire = pvt_wires.read(wire_ptr);
    CircuitNode in_node  = get_node(pvt_nodes, shr_nodes, ghost_nodes, wire.in_loc, wire.in_ptr);
    CircuitNode out_node = get_node(pvt_nodes, shr_nodes, ghost_nodes, wire.out_loc, wire.out_ptr);

    // Solve RLC model iteratively
    float dt = DELTAT;
    const int steps = STEPS;
    float new_v[WIRE_SEGMENTS+1];
    float new_i[WIRE_SEGMENTS];
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      new_i[i] = wire.current[i];
    for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      new_v[i] = wire.voltage[i];
    new_v[WIRE_SEGMENTS] = out_node.voltage;

    for (int j = 0; j < steps; j++)
    {
      // first, figure out the new current from the voltage differential
      // and our inductance:
      // dV = R*I + L*I' ==> I = (dV - L*I')/R
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        new_i[i] = ((new_v[i+1] - new_v[i]) - 
                    (wire.inductance*(new_i[i] - wire.current[i])/dt)) / wire.resistance;
      }
      // Now update the inter-node voltages
      for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      {
        new_v[i+1] = wire.voltage[i] + dt*(new_i[i] - new_i[i+1]) / wire.capacitance;
      }
    }

    // Copy everything back
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      wire.current[i] = new_i[i];
    for (int i = 0; i < WIRE_SEGMENTS-1; i++)
      wire.voltage[i] = new_v[i+1];

    pvt_wires.write(wire_ptr, wire);
  }
#endif
}

void distribute_charge_cpu(CircuitPiece *p,
                           const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::Generic, CircuitWire> pvt_wires = regions[0].get_accessor().typeify<CircuitWire>();
  RegionAccessor<AccessorType::Generic, CircuitNode> pvt_nodes = regions[1].get_accessor().typeify<CircuitNode>();
  RegionAccessor<AccessorType::Generic, CircuitNode> shr_temp = regions[2].get_accessor().typeify<CircuitNode>();
  RegionAccessor<AccessorType::Generic, CircuitNode> ghost_temp = regions[3].get_accessor().typeify<CircuitNode>();
  // Check that we can convert to reduction fold instances
  assert(shr_temp.can_convert<AccessorType::ReductionFold<AccumulateCharge> >());
  assert(ghost_temp.can_convert<AccessorType::ReductionFold<AccumulateCharge> >());
  // Perform the conversion
  RegionAccessor<AccessorType::ReductionFold<AccumulateCharge>, CircuitNode> shr_nodes =
            shr_temp.convert<AccessorType::ReductionFold<AccumulateCharge> >();
  RegionAccessor<AccessorType::ReductionFold<AccumulateCharge>, CircuitNode> ghost_nodes = 
            ghost_temp.convert<AccessorType::ReductionFold<AccumulateCharge> >(); 
  LegionRuntime::HighLevel::IndexIterator itr(p->pvt_wires);
  while (itr.has_next())
  {
    ptr_t wire_ptr = itr.next();
    CircuitWire wire = pvt_wires.read(wire_ptr);

    const float dt = DELTAT; 

    reduce_node<AccumulateCharge>(pvt_nodes,shr_nodes,ghost_nodes,wire.in_loc,wire.in_ptr,-dt * wire.current[0]);
    reduce_node<AccumulateCharge>(pvt_nodes,shr_nodes,ghost_nodes,wire.out_loc,wire.out_ptr,dt* wire.current[WIRE_SEGMENTS-1]);
  }
#endif
}

void update_voltages_cpu(CircuitPiece *p,
                         const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATh
  RegionAccessor<AccessorType::Generic, CircuitNode> pvt_nodes = regions[0].get_accessor().typeify<CircuitNode>();
  IndexIterator pvt_itr(p->pvt_nodes);
  update_region_voltages(p, pvt_nodes, pvt_itr);
  RegionAccessor<AccessorType::Generic, CircuitNode> shr_nodes = regions[1].get_accessor().typeify<CircuitNode>();
  IndexIterator shr_itr(p->shr_nodes);
  update_region_voltages(p, shr_nodes, shr_itr);
#endif
}

