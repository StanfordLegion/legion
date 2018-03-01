/* Copyright 2018 Stanford University
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
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#endif
#include <cmath>

const float AccumulateCharge::identity = 0.0f;

template <>
void AccumulateCharge::apply<true>(LHS &lhs, RHS rhs) 
{
  lhs += rhs;
}

template<>
void AccumulateCharge::apply<false>(LHS &lhs, RHS rhs)
{
  volatile int *target = (volatile int *)&lhs;
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
  volatile int *target = (volatile int *)&rhs1;
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
                             READ_WRITE, EXCLUSIVE, lr_all_wires);
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    rr_out.add_field(FID_CURRENT+i);
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    rr_out.add_field(FID_WIRE_VOLTAGE+i);
  add_region_requirement(rr_out);

  RegionRequirement rr_wires(lp_pvt_wires, 0/*identity*/,
                             READ_ONLY, EXCLUSIVE, lr_all_wires);
  rr_wires.add_field(FID_IN_PTR);
  rr_wires.add_field(FID_OUT_PTR);
  rr_wires.add_field(FID_IN_LOC);
  rr_wires.add_field(FID_OUT_LOC);
  rr_wires.add_field(FID_INDUCTANCE);
  rr_wires.add_field(FID_RESISTANCE);
  rr_wires.add_field(FID_WIRE_CAP);
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

bool CalcNewCurrentsTask::launch_check_fields(Context ctx, Runtime *runtime)
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

static inline float get_node_voltage(const AccessorROfloat &priv,
                                     const AccessorROfloat &shr,
                                     const AccessorROfloat &ghost,
                                     PointerLocation loc, Point<1> ptr)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      return priv[ptr];
    case SHARED_PTR:
      return shr[ptr];
    case GHOST_PTR:
      return ghost[ptr];
    default:
      assert(false);
  }
  return 0.f;
}

#if defined(__AVX512F__)
static inline __m512 get_vec_node_voltage(Point<1> current_wire,
                                          const AccessorROfloat &priv,
                                          const AccessorROfloat &shr,
                                          const AccessorROfloat &ghost,
                                          const AccessorROpoint &ptrs,
                                          const AccessorROloc &locs)
{
  float voltages[16];
  for (int i = 0; i < 16; i++)
  {
    const Point<1> node_ptr = ptrs[current_wire+i];
    PointerLocation loc = locs[current_wire+i];
    switch (loc)
    {
      case PRIVATE_PTR:
        voltages[i] = priv[node_ptr];
        break;
      case SHARED_PTR:
        voltages[i] = shr[node_ptr];
        break;
      case GHOST_PTR:
        voltages[i] = ghost[node_ptr];
        break;
      default:
        assert(false);
    }
  }
  return _mm512_set_ps(voltages[15],voltages[14],voltages[13],voltages[12],
                       voltages[11],voltages[10],voltages[9],voltages[8],
                       voltages[7],voltages[6],voltages[5],voltages[4],
                       voltages[3],voltages[2],voltages[1],voltages[0]);
}

#elif defined(__AVX__)
static inline __m256 get_vec_node_voltage(Point<1> current_wire,
                                          const AccessorROfloat &priv,
                                          const AccessorROfloat &shr,
                                          const AccessorROfloat &ghost,
                                          const AccessorROpoint &ptrs,
                                          const AccessorROloc &locs)
{
  float voltages[8];
  for (int i = 0; i < 8; i++)
  {
    const Point<1> node_ptr = ptrs[current_wire+i];
    PointerLocation loc = locs[current_wire+i];
    switch (loc)
    {
      case PRIVATE_PTR:
        voltages[i] = priv[node_ptr];
        break;
      case SHARED_PTR:
        voltages[i] = shr[node_ptr];
        break;
      case GHOST_PTR:
        voltages[i] = ghost[node_ptr];
        break;
      default:
        assert(false);
    }
  }
  return _mm256_set_ps(voltages[7],voltages[6],voltages[5],voltages[4],
                       voltages[3],voltages[2],voltages[1],voltages[0]);
}

#elif defined(__SSE__)
static inline __m128 get_vec_node_voltage(Point<1> current_wire,
                                          const AccessorROfloat &priv,
                                          const AccessorROfloat &shr,
                                          const AccessorROfloat &ghost,
                                          const AccessorROpoint &ptrs,
                                          const AccessorROloc &locs)
{
  float voltages[4];
  for (int i = 0; i < 4; i++)
  {
    const Point<1> node_ptr = ptrs[current_wire+i];
    PointerLocation loc = locs[current_wire+i];
    switch (loc)
    {
      case PRIVATE_PTR:
        voltages[i] = priv[node_ptr];
        break;
      case SHARED_PTR:
        voltages[i] = shr[node_ptr];
        break;
      case GHOST_PTR:
        voltages[i] = ghost[node_ptr];
        break;
      default:
        assert(false);
    }
  }
  return _mm_set_ps(voltages[3],voltages[2],voltages[1],voltages[0]);
}
#endif

/*static*/
void CalcNewCurrentsTask::cpu_base_impl(const CircuitPiece &piece,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context ctx, Runtime* rt)
{
#ifndef DISABLE_MATH
  AccessorRWfloat fa_current[WIRE_SEGMENTS];
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    fa_current[i] = AccessorRWfloat(regions[0], FID_CURRENT+i);
  AccessorRWfloat fa_voltage[WIRE_SEGMENTS-1];
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    fa_voltage[i] = AccessorRWfloat(regions[0], FID_WIRE_VOLTAGE+i);

  const AccessorROpoint fa_in_ptr(regions[1], FID_IN_PTR);
  const AccessorROpoint fa_out_ptr(regions[1], FID_OUT_PTR);
  const AccessorROloc fa_in_loc(regions[1], FID_IN_LOC);
  const AccessorROloc fa_out_loc(regions[1], FID_OUT_LOC);
  const AccessorROfloat fa_inductance(regions[1], FID_INDUCTANCE);
  const AccessorROfloat fa_resistance(regions[1], FID_RESISTANCE);
  const AccessorROfloat fa_wire_cap(regions[1], FID_WIRE_CAP);

  const AccessorROfloat fa_pvt_voltage(regions[2], FID_NODE_VOLTAGE);
  const AccessorROfloat fa_shr_voltage(regions[3], FID_NODE_VOLTAGE);
  const AccessorROfloat fa_ghost_voltage(regions[4], FID_NODE_VOLTAGE);

  unsigned index = 0;
  const int steps = piece.steps;
#if defined(__AVX512F__)
  // using AVX512F intrinsics, we can work on wires 16-at-a-time
  {
    __m512 temp_v[WIRE_SEGMENTS+1];
    __m512 temp_i[WIRE_SEGMENTS];
    __m512 old_i[WIRE_SEGMENTS];
    __m512 old_v[WIRE_SEGMENTS-1];
    __m512 dt = _mm512_set1_ps(piece.dt);
    __m512 recip_dt = _mm512_set1_ps(1.0/piece.dt);
    while ((index+15) < piece.num_wires)
    {
      // We can do pointer math!
      const Point<1> current_wire = piece.first_wire+index;
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = _mm512_load_ps(fa_current[i].ptr(current_wire));
        old_i[i] = temp_i[i];
      }
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = _mm512_load_ps(fa_voltage[i].ptr(current_wire));
        old_v[i] = temp_v[i+1];
      }

      // Pin the outer voltages to the node voltages
      temp_v[0] = get_vec_node_voltage(current_wire, fa_pvt_voltage,
                                       fa_shr_voltage, fa_ghost_voltage,
                                       fa_in_ptr, fa_in_loc);
      temp_v[WIRE_SEGMENTS] = get_vec_node_voltage(current_wire, fa_pvt_voltage,
                                       fa_shr_voltage, fa_ghost_voltage,
                                       fa_out_ptr, fa_out_loc);
      __m512 inductance = _mm512_load_ps(fa_inductance.ptr(current_wire));
      __m512 recip_resistance = _mm512_rcp14_ps(_mm512_load_ps(fa_resistance.ptr(current_wire)));
      __m512 recip_capacitance = _mm512_rcp14_ps(_mm512_load_ps(fa_wire_cap.ptr(current_wire)));
      for (int j = 0; j < steps; j++)
      {
        for (int i = 0; i < WIRE_SEGMENTS; i++)
        {
          __m512 dv = _mm512_sub_ps(temp_v[i+1],temp_v[i]);
          __m512 di = _mm512_sub_ps(temp_i[i],old_i[i]);
          __m512 vol = _mm512_sub_ps(dv,_mm512_mul_ps(_mm512_mul_ps(inductance,di),recip_dt));
          temp_i[i] = _mm512_mul_ps(vol,recip_resistance);
        }
        for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        {
          __m512 dq = _mm512_mul_ps(dt,_mm512_sub_ps(temp_i[i],temp_i[i+1]));
          temp_v[i+1] = _mm512_add_ps(old_v[i],_mm512_mul_ps(dq,recip_capacitance));
        }
      }
      // Write out the results
      for (int i = 0; i < WIRE_SEGMENTS; i++)
        _mm512_stream_ps(fa_current[i].ptr(current_wire),temp_i[i]);
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        _mm512_stream_ps(fa_voltage[i].ptr(current_wire),temp_v[i+1]);
      // Update the index
      index += 16;
    }
  }

#elif defined(__AVX__)
  // using AVX intrinsics, we can work on wires 8-at-a-time
  {
    __m256 temp_v[WIRE_SEGMENTS+1];
    __m256 temp_i[WIRE_SEGMENTS];
    __m256 old_i[WIRE_SEGMENTS];
    __m256 old_v[WIRE_SEGMENTS-1];
    __m256 dt = _mm256_set1_ps(piece.dt);
    __m256 recip_dt = _mm256_set1_ps(1.0/piece.dt);
    while ((index+7) < piece.num_wires)
    {
      // We can do pointer math!
      const Point<1> current_wire = piece.first_wire+index;
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = _mm256_load_ps(fa_current[i].ptr(current_wire));
        old_i[i] = temp_i[i];
      }
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = _mm256_load_ps(fa_voltage[i].ptr(current_wire));
        old_v[i] = temp_v[i+1];
      }

      // Pin the outer voltages to the node voltages
      temp_v[0] = get_vec_node_voltage(current_wire, fa_pvt_voltage,
                                       fa_shr_voltage, fa_ghost_voltage,
                                       fa_in_ptr, fa_in_loc);
      temp_v[WIRE_SEGMENTS] = get_vec_node_voltage(current_wire, fa_pvt_voltage,
                                       fa_shr_voltage, fa_ghost_voltage,
                                       fa_out_ptr, fa_out_loc);
      __m256 inductance = _mm256_load_ps(fa_inductance.ptr(current_wire));
      __m256 recip_resistance = _mm256_rcp_ps(_mm256_load_ps(fa_resistance.ptr(current_wire)));
      __m256 recip_capacitance = _mm256_rcp_ps(_mm256_load_ps(fa_wire_cap.ptr(current_wire)));
      for (int j = 0; j < steps; j++)
      {
        for (int i = 0; i < WIRE_SEGMENTS; i++)
        {
          __m256 dv = _mm256_sub_ps(temp_v[i+1],temp_v[i]);
          __m256 di = _mm256_sub_ps(temp_i[i],old_i[i]);
          __m256 vol = _mm256_sub_ps(dv,_mm256_mul_ps(_mm256_mul_ps(inductance,di),recip_dt));
          temp_i[i] = _mm256_mul_ps(vol,recip_resistance);
        }
        for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        {
          __m256 dq = _mm256_mul_ps(dt,_mm256_sub_ps(temp_i[i],temp_i[i+1]));
          temp_v[i+1] = _mm256_add_ps(old_v[i],_mm256_mul_ps(dq,recip_capacitance));
        }
      }
      // Write out the results
      for (int i = 0; i < WIRE_SEGMENTS; i++)
        _mm256_stream_ps(fa_current[i].ptr(current_wire),temp_i[i]);
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        _mm256_stream_ps(fa_voltage[i].ptr(current_wire),temp_v[i+1]);
      // Update the index
      index += 8;
    }
  }

#elif defined(__SSE__)
  // using SSE intrinsics, we can work on wires 4-at-a-time
  {
    __m128 temp_v[WIRE_SEGMENTS+1];
    __m128 temp_i[WIRE_SEGMENTS];
    __m128 old_i[WIRE_SEGMENTS];
    __m128 old_v[WIRE_SEGMENTS-1];
    __m128 dt = _mm_set1_ps(piece.dt);
    __m128 recip_dt = _mm_set1_ps(1.0/piece.dt);
    while ((index+3) < piece.num_wires)
    {
      // We can do pointer math!
      const Point<1> current_wire = piece.first_wire+index;
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = _mm_load_ps(fa_current[i].ptr(current_wire));
        old_i[i] = temp_i[i];
      }
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = _mm_load_ps(fa_voltage[i].ptr(current_wire));
        old_v[i] = temp_v[i+1];
      }

      // Pin the outer voltages to the node voltages
      temp_v[0] = get_vec_node_voltage(current_wire, fa_pvt_voltage,
                                       fa_shr_voltage, fa_ghost_voltage,
                                       fa_in_ptr, fa_in_loc);
      temp_v[WIRE_SEGMENTS] = get_vec_node_voltage(current_wire, fa_pvt_voltage,
                                       fa_shr_voltage, fa_ghost_voltage,
                                       fa_out_ptr, fa_out_loc);
      __m128 inductance = _mm_load_ps(fa_inductance.ptr(current_wire));
      __m128 recip_resistance = _mm_rcp_ps(_mm_load_ps(fa_resistance.ptr(current_wire)));
      __m128 recip_capacitance = _mm_rcp_ps(_mm_load_ps(fa_wire_cap.ptr(current_wire)));
      for (int j = 0; j < steps; j++)
      {
        for (int i = 0; i < WIRE_SEGMENTS; i++)
        {
          __m128 dv = _mm_sub_ps(temp_v[i+1],temp_v[i]);
          __m128 di = _mm_sub_ps(temp_i[i],old_i[i]);
          __m128 vol = _mm_sub_ps(dv,_mm_mul_ps(_mm_mul_ps(inductance,di),recip_dt));
          temp_i[i] = _mm_mul_ps(vol,recip_resistance);
        }
        for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        {
          __m128 dq = _mm_mul_ps(dt,_mm_sub_ps(temp_i[i],temp_i[i+1]));
          temp_v[i+1] = _mm_add_ps(old_v[i],_mm_mul_ps(dq,recip_capacitance));
        }
      }
      // Write out the results
      for (int i = 0; i < WIRE_SEGMENTS; i++)
        _mm_stream_ps(fa_current[i].ptr(current_wire),temp_i[i]);
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        _mm_stream_ps(fa_voltage[i].ptr(current_wire),temp_v[i+1]);
      // Update the index
      index += 4;
    }
  }
#endif

  float temp_v[WIRE_SEGMENTS+1];
  float temp_i[WIRE_SEGMENTS];
  float old_i[WIRE_SEGMENTS];
  float old_v[WIRE_SEGMENTS-1];
  const float dt = piece.dt;
  const float recip_dt = 1.0f / dt;
  for (unsigned w = index; w < piece.num_wires; w++) 
  {
    const Point<1> wire_ptr = piece.first_wire + w;
    for (int i = 0; i < WIRE_SEGMENTS; i++)
    {
      temp_i[i] = fa_current[i][wire_ptr];
      old_i[i] = temp_i[i];
    }
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    {
      temp_v[i+1] = fa_voltage[i][wire_ptr];
      old_v[i] = temp_v[i+1];
    }

    // Pin the outer voltages to the node voltages
    Point<1> in_ptr = fa_in_ptr[wire_ptr];
    PointerLocation in_loc = fa_in_loc[wire_ptr];
    temp_v[0] = 
      get_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, in_loc, in_ptr);
    Point<1> out_ptr = fa_out_ptr[wire_ptr];
    PointerLocation out_loc = fa_out_loc[wire_ptr];
    temp_v[WIRE_SEGMENTS] = 
      get_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, out_loc, out_ptr);

    // Solve the RLC model iteratively
    float inductance = fa_inductance[wire_ptr];
    float recip_resistance = 1.0f / fa_resistance[wire_ptr];
    float recip_capacitance = 1.0f / fa_wire_cap[wire_ptr];
    for (int j = 0; j < steps; j++)
    {
      // first, figure out the new current from the voltage differential
      // and our inductance:
      // dV = R*I + L*I' ==> I = (dV - L*I')/R
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = ((temp_v[i+1] - temp_v[i]) - 
                     (inductance * (temp_i[i] - old_i[i]) * recip_dt)) * recip_resistance;
      }
      // Now update the inter-node voltages
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = old_v[i] + dt * (temp_i[i] - temp_i[i+1]) * recip_capacitance;
      }
    }

    // Write out the results
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      fa_current[i][wire_ptr] = temp_i[i];
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      fa_voltage[i][wire_ptr] = temp_v[i+1];
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
  rr_wires.add_field(FID_IN_PTR);
  rr_wires.add_field(FID_OUT_PTR);
  rr_wires.add_field(FID_IN_LOC);
  rr_wires.add_field(FID_OUT_LOC);
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

bool DistributeChargeTask::launch_check_fields(Context ctx, Runtime *runtime)
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

typedef ReductionAccessor<AccumulateCharge,false/*exclusive*/,1,coord_t,
                          Realm::AffineAccessor<float,1,coord_t> > AccessorRDfloat;

static inline void reduce_node(const AccessorRWfloat &priv,
                               const AccessorRDfloat &shr,
                               const AccessorRDfloat &ghost,
                               PointerLocation loc, Point<1> ptr, float value)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      AccumulateCharge::apply<true/*exclusive*/>(priv[ptr], value);
      break;
    case SHARED_PTR:
      shr[ptr] <<= value;
      break;
    case GHOST_PTR:
      ghost[ptr] <<= value;
      break;
    default:
      assert(false);
  }
}

/*static*/
void DistributeChargeTask::cpu_base_impl(const CircuitPiece &p,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context ctx, Runtime* rt)
{
#ifndef DISABLE_MATH
  const AccessorROpoint fa_in_ptr(regions[0], FID_IN_PTR);
  const AccessorROpoint fa_out_ptr(regions[0], FID_OUT_PTR);
  const AccessorROloc fa_in_loc(regions[0], FID_IN_LOC);
  const AccessorROloc fa_out_loc(regions[0], FID_OUT_LOC);
  const AccessorROfloat fa_in_current(regions[0], FID_CURRENT);
  const AccessorROfloat fa_out_current(regions[0], FID_CURRENT+WIRE_SEGMENTS-1);
  const AccessorRWfloat fa_pvt_charge(regions[1], FID_CHARGE);
  const AccessorRDfloat fa_shr_charge(regions[2], FID_CHARGE, REDUCE_ID);
  const AccessorRDfloat fa_ghost_charge(regions[3], FID_CHARGE, REDUCE_ID);

  const float dt = p.dt;
  for (unsigned i = 0; i < p.num_wires; i++)
  {
    const Point<1> wire_ptr(p.first_wire + i);
#ifdef DEBUG_MATH
    printf("DC: %d = %f->(%d,%d), %f->(%d,%d)\n",
           wire_ptr,
           fa_in_current[wire_ptr],
           fa_in_ptr[wire_ptr], fa_in_loc[wire_ptr],
           fa_out_current[wire_ptr],
           fa_out_ptr[wire_ptr], fa_out_loc[wire_ptr]);
#endif
    float in_current = -dt * fa_in_current[wire_ptr];
    float out_current = dt * fa_out_current[wire_ptr];
    Point<1> in_ptr = fa_in_ptr[wire_ptr];
    Point<1> out_ptr = fa_out_ptr[wire_ptr];
    PointerLocation in_loc = fa_in_loc[wire_ptr];
    PointerLocation out_loc = fa_out_loc[wire_ptr];

    reduce_node(fa_pvt_charge, fa_shr_charge, fa_ghost_charge,
                in_loc, in_ptr, in_current);
    reduce_node(fa_pvt_charge, fa_shr_charge, fa_ghost_charge,
                out_loc, out_ptr, out_current);
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
  rr_private_in.add_field(FID_NODE_CAP);
  rr_private_in.add_field(FID_LEAKAGE);
  add_region_requirement(rr_private_in);

  RegionRequirement rr_shared_in(lp_shr_nodes, 0/*identity*/,
                                 READ_ONLY, EXCLUSIVE, lr_all_nodes);
  rr_shared_in.add_field(FID_NODE_CAP);
  rr_shared_in.add_field(FID_LEAKAGE);
  add_region_requirement(rr_shared_in);

  RegionRequirement rr_locator(lp_node_locations, 0/*identity*/,
                               READ_ONLY, EXCLUSIVE, lr_node_locator);
  rr_locator.add_field(FID_LOCATOR);
  add_region_requirement(rr_locator);
}

/*static*/
const char * const UpdateVoltagesTask::TASK_NAME = "update_voltages";

bool UpdateVoltagesTask::launch_check_fields(Context ctx, Runtime *runtime)
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

static inline void update_voltages(LogicalRegion lr,
                                   const AccessorRWfloat &fa_voltage,
                                   const AccessorRWfloat &fa_charge,
                                   const AccessorROfloat &fa_cap,
                                   const AccessorROfloat &fa_leakage,
                                   Context ctx, Runtime* rt)
{
  for (PointInDomainIterator<1> itr(
        rt->get_index_space_domain(lr.get_index_space())); itr(); itr++)
  {
    float voltage = fa_voltage[*itr];
    float charge = fa_charge[*itr];
    float capacitance = fa_cap[*itr];
    float leakage = fa_leakage[*itr];
    voltage += charge / capacitance;
    voltage *= (1.f - leakage);
    fa_voltage[*itr] = voltage;
    // Reset the charge for the next iteration
    fa_charge[*itr] = 0.f;
  }
}

/*static*/
void UpdateVoltagesTask::cpu_base_impl(const CircuitPiece &p,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, Runtime* rt)
{
#ifndef DISABLE_MATH
  const AccessorRWfloat fa_pvt_voltage(regions[0], FID_NODE_VOLTAGE);
  const AccessorRWfloat fa_pvt_charge(regions[0], FID_CHARGE);

  const AccessorRWfloat fa_shr_voltage(regions[1], FID_NODE_VOLTAGE);
  const AccessorRWfloat fa_shr_charge(regions[1], FID_CHARGE);

  const AccessorROfloat fa_pvt_cap(regions[2], FID_NODE_CAP);
  const AccessorROfloat fa_pvt_leakage(regions[2], FID_LEAKAGE);

  const AccessorROfloat fa_shr_cap(regions[3], FID_NODE_CAP);
  const AccessorROfloat fa_shr_leakage(regions[3], FID_LEAKAGE);
  // Don't need this for the CPU version
  // const AccessorROloc fa_location(regions[4], FID_LOCATOR);

  update_voltages(p.pvt_nodes, fa_pvt_voltage, fa_pvt_charge, 
                  fa_pvt_cap, fa_pvt_leakage, ctx, rt);
  update_voltages(p.shr_nodes, fa_shr_voltage, fa_shr_charge, 
                  fa_shr_cap, fa_shr_leakage, ctx, rt);
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

bool CheckTask::dispatch(Context ctx, Runtime *runtime, bool success)
{
  FutureMap fm = runtime->execute_index_space(ctx, *this);
  fm.wait_all_results();
  Rect<1> launch_array = launch_domain;
  for (PointInRectIterator<1> pir(launch_array); pir(); pir++)
    success = fm.get_result<bool>(*pir) && success;
  return success;
}

/*static*/
bool CheckTask::cpu_impl(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  const AccessorROfloat fa_check(regions[0], task->regions[0].instance_fields[0]);
  LogicalRegion lr = task->regions[0].region;
  bool success = true;
  for (PointInDomainIterator<1> itr(
        runtime->get_index_space_domain(lr.get_index_space())); itr(); itr++)
  {
    float value = fa_check[*itr];
    if (std::isnan(value))
      success = false;
  }
  return success;
}

/*static*/
void CheckTask::register_task(void)
{
  TaskVariantRegistrar registrar(CheckTask::TASK_ID, CheckTask::TASK_NAME);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(CheckTask::LEAF);
  Runtime::preregister_task_variant<bool, cpu_impl>(registrar, CheckTask::TASK_NAME);
}

