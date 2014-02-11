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
  lhs += rhs;
}

template<>
void AccumulateCharge::apply<false>(LHS &lhs, RHS rhs)
{
  int *target = (int *)&lhs;
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

template <>
void AccumulateCharge::fold<true>(RHS &rhs1, RHS rhs2) 
{
  rhs1 += rhs2;
}

template<>
void AccumulateCharge::fold<false>(RHS &rhs1, RHS rhs2)
{
  int *target = (int *)&rhs1;
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs2;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

CalcNewCurrentsTask::CalcNewCurrentsTask(LogicalPartition lp_pvt_wires,
                                         LogicalPartition lp_pvt_nodes,
                                         LogicalPartition lp_shr_nodes,
                                         LogicalPartition lp_ghost_nodes,
                                         LogicalRegion lr_all_wires,
                                         LogicalRegion lr_all_nodes,
                                         const Domain &launch_domain,
                                         const ArgumentMap &arg_map)
 : IndexLauncher(CalcNewCurrentsTask::TASK_ID, launch_domain, TaskArgument(), arg_map,
                 Predicate::TRUE_PRED, false/*must*/, CalcNewCurrentsTask::MAPPER_ID)
{
  RegionRequirement rr_out(lp_pvt_wires, 0/*identity*/, 
                             WRITE_DISCARD, EXCLUSIVE, lr_all_wires);
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    rr_out.add_field(FID_CURRENT+i);
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    rr_out.add_field(FID_WIRE_VOLTAGE+i);
  add_region_requirement(rr_out);

  RegionRequirement rr_wires(lp_pvt_wires, 0/*identity*/,
                             READ_ONLY, EXCLUSIVE, lr_all_wires);
  rr_wires.add_field(FID_WIRE_PROP);
  add_region_requirement(rr_wires);

  RegionRequirement rr_private(lp_pvt_nodes, 0/*identity*/,
                               READ_ONLY, EXCLUSIVE, lr_all_nodes);
  rr_private.add_field(FID_NODE_VOLTAGE);
  add_region_requirement(rr_private);

  RegionRequirement rr_shared(lp_shr_nodes, 0/*identity*/,
                              READ_ONLY, EXCLUSIVE, lr_all_nodes);
  rr_shared.add_field(FID_NODE_VOLTAGE);
  add_region_requirement(rr_shared);

  RegionRequirement rr_ghost(lp_ghost_nodes, 0/*identity*/,
                             READ_ONLY, EXCLUSIVE, lr_all_nodes);
  rr_ghost.add_field(FID_NODE_VOLTAGE);
  add_region_requirement(rr_ghost);
}

/*static*/ const char * const CalcNewCurrentsTask::TASK_NAME = "calc_new_currents";

bool CalcNewCurrentsTask::launch_check_fields(Context ctx, HighLevelRuntime *runtime)
{
  const RegionRequirement &req = region_requirements[0];
  bool success = true;
  for (int i = 0; i < WIRE_SEGMENTS; i++)
  {
    CheckTask launcher(req.partition, req.parent, FID_CURRENT+i, launch_domain, argument_map); 
    success = launcher.dispatch(ctx, runtime, success); 
  }
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
  {
    CheckTask launcher(req.partition, req.parent, FID_WIRE_VOLTAGE+i, launch_domain, argument_map);
    success = launcher.dispatch(ctx, runtime, success);
  }
  return success;
}

template<typename AT>
static inline float get_node_voltage(const RegionAccessor<AT,float> &priv,
                                     const RegionAccessor<AT,float> &shr,
                                     const RegionAccessor<AT,float> &ghost,
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
  return 0.f;
}

/*static*/
void CalcNewCurrentsTask::cpu_base_impl(const CircuitPiece &p,
                                        const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::Generic, float> fa_current[WIRE_SEGMENTS];
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    fa_current[i] = regions[0].get_field_accessor(FID_CURRENT+i).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_voltage[WIRE_SEGMENTS-1];
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    fa_voltage[i] = regions[0].get_field_accessor(FID_WIRE_VOLTAGE+i).typeify<float>();
  RegionAccessor<AccessorType::Generic, WireProperties> fa_prop = 
    regions[1].get_field_accessor(FID_WIRE_PROP).typeify<WireProperties>();
  RegionAccessor<AccessorType::Generic, float> fa_pvt_voltage = 
    regions[2].get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_voltage =
    regions[3].get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_ghost_voltage = 
    regions[4].get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();

  LegionRuntime::HighLevel::IndexIterator itr(p.pvt_wires);
  float temp_v[WIRE_SEGMENTS+1];
  float temp_i[WIRE_SEGMENTS];
  float old_i[WIRE_SEGMENTS];
  float old_v[WIRE_SEGMENTS-1];
  while (itr.has_next())
  {
    ptr_t wire_ptr = itr.next();
    const WireProperties prop = fa_prop.read(wire_ptr);
    const float dt = DELTAT;
    const int steps = STEPS;

    for (int i = 0; i < WIRE_SEGMENTS; i++)
    {
      temp_i[i] = fa_current[i].read(wire_ptr);
      old_i[i] = temp_i[i];
    }
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    {
      temp_v[i+1] = fa_voltage[i].read(wire_ptr);
      old_v[i] = temp_v[i+1];
    }

    // Pin the outer voltages to the node voltages
    temp_v[0] = 
      get_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, prop.in_loc, prop.in_ptr);
    temp_v[WIRE_SEGMENTS] = 
      get_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, prop.out_loc, prop.out_ptr);

    // Solve the RLC model iteratively
    for (int j = 0; j < steps; j++)
    {
      // first, figure out the new current from the voltage differential
      // and our inductance:
      // dV = R*I + L*I' ==> I = (dV - L*I')/R
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = ((temp_v[i+1] - temp_v[i]) - 
                     (prop.inductance * (temp_i[i] - old_i[i]) / dt)) / prop.resistance; 
      }
      // Now update the inter-node voltages
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = old_v[i] + dt * (temp_i[i] - temp_i[i+1]) / prop.capacitance;
      }
    }

    // Write out the results
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      fa_current[i].write(wire_ptr, temp_i[i]);
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      fa_voltage[i].write(wire_ptr, temp_v[i+1]);
  }
#endif
}

DistributeChargeTask::DistributeChargeTask(LogicalPartition lp_pvt_wires,
                                           LogicalPartition lp_pvt_nodes,
                                           LogicalPartition lp_shr_nodes,
                                           LogicalPartition lp_ghost_nodes,
                                           LogicalRegion lr_all_wires,
                                           LogicalRegion lr_all_nodes,
                                           const Domain &launch_domain,
                                           const ArgumentMap &arg_map)
 : IndexLauncher(DistributeChargeTask::TASK_ID, launch_domain, TaskArgument(), arg_map,
                 Predicate::TRUE_PRED, false/*must*/, DistributeChargeTask::MAPPER_ID)
{
  RegionRequirement rr_wires(lp_pvt_wires, 0/*identity*/,
                             READ_ONLY, EXCLUSIVE, lr_all_wires);
  rr_wires.add_field(FID_WIRE_PROP);
  rr_wires.add_field(FID_CURRENT);
  rr_wires.add_field(FID_CURRENT+WIRE_SEGMENTS-1);
  add_region_requirement(rr_wires);

  RegionRequirement rr_private(lp_pvt_nodes, 0/*identity*/,
                               READ_WRITE, EXCLUSIVE, lr_all_nodes);
  rr_private.add_field(FID_CHARGE);
  add_region_requirement(rr_private);

  RegionRequirement rr_shared(lp_shr_nodes, 0/*identity*/,
                              REDUCE_ID, SIMULTANEOUS, lr_all_nodes);
  rr_shared.add_field(FID_CHARGE);
  add_region_requirement(rr_shared);

  RegionRequirement rr_ghost(lp_ghost_nodes, 0/*identity*/,
                             REDUCE_ID, SIMULTANEOUS, lr_all_nodes);
  rr_ghost.add_field(FID_CHARGE);
  add_region_requirement(rr_ghost);
}

/*static*/ const char * const DistributeChargeTask::TASK_NAME = "distribute_charge";

bool DistributeChargeTask::launch_check_fields(Context ctx, HighLevelRuntime *runtime)
{
  bool success = true;
  for (unsigned idx = 1; idx < 4; idx++)
  {
    const RegionRequirement &req = region_requirements[idx];
    CheckTask launcher(req.partition, req.parent, FID_CHARGE, launch_domain, argument_map);
    success = launcher.dispatch(ctx, runtime, success);
  }
  return success;
}

template<typename REDOP, typename AT1, typename AT2>
static inline void reduce_node(const RegionAccessor<AT1,typename REDOP::LHS> &priv,
                               const RegionAccessor<AT2,typename REDOP::LHS> &shr,
                               const RegionAccessor<AT2,typename REDOP::LHS> &ghost,
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

/*static*/
void DistributeChargeTask::cpu_base_impl(const CircuitPiece &p,
                                         const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::Generic, WireProperties> fa_prop = 
    regions[0].get_field_accessor(FID_WIRE_PROP).typeify<WireProperties>();
  RegionAccessor<AccessorType::Generic, float> fa_in_current = 
    regions[0].get_field_accessor(FID_CURRENT).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_out_current = 
    regions[0].get_field_accessor(FID_CURRENT+WIRE_SEGMENTS-1).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_pvt_charge = 
    regions[1].get_field_accessor(FID_CHARGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_temp = 
    regions[2].get_accessor().typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_ghost_temp =
    regions[3].get_accessor().typeify<float>();
  // Check that we can convert to reduction fold instances
  assert(fa_shr_temp.can_convert<AccessorType::ReductionFold<AccumulateCharge> >());
  assert(fa_ghost_temp.can_convert<AccessorType::ReductionFold<AccumulateCharge> >());
  // Perform the conversion
  RegionAccessor<AccessorType::ReductionFold<AccumulateCharge>, float> fa_shr_charge = 
    fa_shr_temp.convert<AccessorType::ReductionFold<AccumulateCharge> >();
  RegionAccessor<AccessorType::ReductionFold<AccumulateCharge>, float> fa_ghost_charge = 
    fa_ghost_temp.convert<AccessorType::ReductionFold<AccumulateCharge> >();

  LegionRuntime::HighLevel::IndexIterator itr(p.pvt_wires);
  while (itr.has_next())
  {
    ptr_t wire_ptr = itr.next();
    const WireProperties prop = fa_prop.read(wire_ptr);
    const float dt = DELTAT;
    float in_current = -dt * fa_in_current.read(wire_ptr);
    float out_current = dt * fa_out_current.read(wire_ptr);

    reduce_node<AccumulateCharge>(fa_pvt_charge, fa_shr_charge, fa_ghost_charge,
                                  prop.in_loc, prop.in_ptr, in_current);
    reduce_node<AccumulateCharge>(fa_pvt_charge, fa_shr_charge, fa_ghost_charge,
                                  prop.out_loc, prop.out_ptr, out_current);
  }
#endif
}


UpdateVoltagesTask::UpdateVoltagesTask(LogicalPartition lp_pvt_nodes,
                                       LogicalPartition lp_shr_nodes,
                                       LogicalPartition lp_node_locations,
                                       LogicalRegion lr_all_nodes,
                                       LogicalRegion lr_node_locator,
                                       const Domain &launch_domain,
                                       const ArgumentMap &arg_map)
 : IndexLauncher(UpdateVoltagesTask::TASK_ID, launch_domain, TaskArgument(), arg_map,
                 Predicate::TRUE_PRED, false/*must*/, UpdateVoltagesTask::MAPPER_ID)
{
  RegionRequirement rr_private_out(lp_pvt_nodes, 0/*identity*/,
                               READ_WRITE, EXCLUSIVE, lr_all_nodes);
  rr_private_out.add_field(FID_NODE_VOLTAGE);
  rr_private_out.add_field(FID_CHARGE);
  add_region_requirement(rr_private_out);

  RegionRequirement rr_shared_out(lp_shr_nodes, 0/*identity*/,
                                  READ_WRITE, EXCLUSIVE, lr_all_nodes);
  rr_shared_out.add_field(FID_NODE_VOLTAGE);
  rr_shared_out.add_field(FID_CHARGE);
  add_region_requirement(rr_shared_out);

  RegionRequirement rr_private_in(lp_pvt_nodes, 0/*identity*/,
                                  READ_ONLY, EXCLUSIVE, lr_all_nodes);
  rr_private_in.add_field(FID_NODE_PROP);
  add_region_requirement(rr_private_in);

  RegionRequirement rr_shared_in(lp_shr_nodes, 0/*identity*/,
                                 READ_ONLY, EXCLUSIVE, lr_all_nodes);
  rr_shared_in.add_field(FID_NODE_PROP);
  add_region_requirement(rr_shared_in);

  RegionRequirement rr_locator(lp_node_locations, 0/*identity*/,
                               READ_ONLY, EXCLUSIVE, lr_node_locator);
  rr_locator.add_field(FID_LOCATOR);
  add_region_requirement(rr_locator);
}

/*static*/
const char * const UpdateVoltagesTask::TASK_NAME = "update_voltages";

bool UpdateVoltagesTask::launch_check_fields(Context ctx, HighLevelRuntime *runtime)
{
  bool success = true;
  const RegionRequirement &req = region_requirements[0]; 
  {
    CheckTask launcher(req.partition, req.parent, FID_NODE_VOLTAGE, launch_domain, argument_map);
    success = launcher.dispatch(ctx, runtime, success);
  }
  {
    CheckTask launcher(req.partition, req.parent, FID_CHARGE, launch_domain, argument_map);
    success = launcher.dispatch(ctx, runtime, success);
  }
  return success;
}

template<typename AT>
static inline void update_voltages(LogicalRegion lr,
                                   const RegionAccessor<AT,float> &fa_voltage,
                                   const RegionAccessor<AT,float> &fa_charge,
                                   const RegionAccessor<AT,NodeProperties> &fa_prop)
{
  IndexIterator itr(lr);
  while (itr.has_next())
  {
    ptr_t node_ptr = itr.next();
    float voltage = fa_voltage.read(node_ptr);
    float charge = fa_charge.read(node_ptr);
    const NodeProperties prop = fa_prop.read(node_ptr);
    voltage += charge / prop.capacitance;
    voltage *= (1.f - prop.leakage);
    fa_voltage.write(node_ptr, voltage);
    fa_charge.write(node_ptr, 0.f);
  }
}

/*static*/
void UpdateVoltagesTask::cpu_base_impl(const CircuitPiece &p,
                                       const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::Generic, float> fa_pvt_voltage = 
    regions[0].get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_pvt_charge = 
    regions[0].get_field_accessor(FID_CHARGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_voltage = 
    regions[1].get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_charge = 
    regions[1].get_field_accessor(FID_CHARGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, NodeProperties> fa_pvt_prop = 
    regions[2].get_field_accessor(FID_NODE_PROP).typeify<NodeProperties>();
  RegionAccessor<AccessorType::Generic, NodeProperties> fa_shr_prop = 
    regions[3].get_field_accessor(FID_NODE_PROP).typeify<NodeProperties>();
  //RegionAccessor<AccessorType::Generic, PointerLocation> fa_location = 
  //  regions[4].get_field_accessor(FID_LOCATOR).typeify<PointerLocation>();

  update_voltages(p.pvt_nodes, fa_pvt_voltage, fa_pvt_charge, fa_pvt_prop);
  update_voltages(p.shr_nodes, fa_shr_voltage, fa_shr_charge, fa_shr_prop);
#endif
}

CheckTask::CheckTask(LogicalPartition lp,
                     LogicalRegion lr,
                     FieldID fid,
                     const Domain &launch_domain,
                     const ArgumentMap &arg_map)
 : IndexLauncher(CheckTask::TASK_ID, launch_domain, TaskArgument(), arg_map,
                 Predicate::TRUE_PRED, false/*must*/, CheckTask::MAPPER_ID)
{
  RegionRequirement rr_check(lp, 0/*identity*/, READ_ONLY, EXCLUSIVE, lr);
  rr_check.add_field(fid);
  add_region_requirement(rr_check);
}

/*static*/
const char * const CheckTask::TASK_NAME = "check_task";

bool CheckTask::dispatch(Context ctx, HighLevelRuntime *runtime, bool success)
{
  FutureMap fm = runtime->execute_index_space(ctx, *this);
  fm.wait_all_results();
  Rect<1> launch_array = launch_domain.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(launch_array); pir; pir++)
    success = fm.get_result<bool>(DomainPoint::from_point<1>(pir.p)) && success;
  return success;
}

/*static*/
bool CheckTask::cpu_impl(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, HighLevelRuntime *runtime)
{
  RegionAccessor<AccessorType::Generic, float> fa_check = 
    regions[0].get_field_accessor(task->regions[0].instance_fields[0]).typeify<float>();
  LogicalRegion lr = task->regions[0].region;
  IndexIterator itr(lr);
  bool success = true;
  while (itr.has_next() && success)
  {
    ptr_t ptr = itr.next();
    float value = fa_check.read(ptr);
    if (isnan(value))
      success = false;
  }
  return success;
}

/*static*/
void CheckTask::register_task(void)
{
  HighLevelRuntime::register_legion_task<bool, cpu_impl>(CheckTask::TASK_ID, Processor::LOC_PROC,
                                                         false/*single*/, true/*index*/,
                                                         CIRCUIT_CPU_LEAF_VARIANT,
                                                         TaskConfigOptions(CheckTask::LEAF),
                                                         CheckTask::TASK_NAME);
}

