/* Copyright 2017 Stanford University
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

#if defined(__AVX512F__)
template<typename AT_VAL, typename AT_PTR>
static inline __m512 get_vec_node_voltage(ptr_t current_wire,
                                          const RegionAccessor<AT_VAL,float> &priv,
                                          const RegionAccessor<AT_VAL,float> &shr,
                                          const RegionAccessor<AT_VAL,float> &ghost,
                                          const RegionAccessor<AT_PTR,ptr_t> &ptrs,
                                          const RegionAccessor<AT_VAL,PointerLocation> &locs)
{
  float voltages[16];
  for (int i = 0; i < 16; i++)
  {
    ptr_t node_ptr = ptrs.read(current_wire+i);
    PointerLocation loc = locs.read(current_wire+i);
    switch (loc)
    {
      case PRIVATE_PTR:
        voltages[i] = priv.read(node_ptr);
        break;
      case SHARED_PTR:
        voltages[i] = shr.read(node_ptr);
        break;
      case GHOST_PTR:
        voltages[i] = ghost.read(node_ptr);
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
template<typename AT_VAL, typename AT_PTR>
static inline __m256 get_vec_node_voltage(ptr_t current_wire,
                                          const RegionAccessor<AT_VAL,float> &priv,
                                          const RegionAccessor<AT_VAL,float> &shr,
                                          const RegionAccessor<AT_VAL,float> &ghost,
                                          const RegionAccessor<AT_PTR,ptr_t> &ptrs,
                                          const RegionAccessor<AT_VAL,PointerLocation> &locs)
{
  float voltages[8];
  for (int i = 0; i < 8; i++)
  {
    ptr_t node_ptr = ptrs.read(current_wire+i);
    PointerLocation loc = locs.read(current_wire+i);
    switch (loc)
    {
      case PRIVATE_PTR:
        voltages[i] = priv.read(node_ptr);
        break;
      case SHARED_PTR:
        voltages[i] = shr.read(node_ptr);
        break;
      case GHOST_PTR:
        voltages[i] = ghost.read(node_ptr);
        break;
      default:
        assert(false);
    }
  }
  return _mm256_set_ps(voltages[7],voltages[6],voltages[5],voltages[4],
                       voltages[3],voltages[2],voltages[1],voltages[0]);
}

#elif defined(__SSE__)
template<typename AT_VAL, typename AT_PTR>
static inline __m128 get_vec_node_voltage(ptr_t current_wire,
                                          const RegionAccessor<AT_VAL,float> &priv,
                                          const RegionAccessor<AT_VAL,float> &shr,
                                          const RegionAccessor<AT_VAL,float> &ghost,
                                          const RegionAccessor<AT_PTR,ptr_t> &ptrs,
                                          const RegionAccessor<AT_VAL,PointerLocation> &locs)
{
  float voltages[4];
  for (int i = 0; i < 4; i++)
  {
    ptr_t node_ptr = ptrs.read(current_wire+i);
    PointerLocation loc = locs.read(current_wire+i);
    switch (loc)
    {
      case PRIVATE_PTR:
        voltages[i] = priv.read(node_ptr);
        break;
      case SHARED_PTR:
        voltages[i] = shr.read(node_ptr);
        break;
      case GHOST_PTR:
        voltages[i] = ghost.read(node_ptr);
        break;
      default:
        assert(false);
    }
  }
  return _mm_set_ps(voltages[3],voltages[2],voltages[1],voltages[0]);
}
#endif

/*static*/
bool CalcNewCurrentsTask::dense_calc_new_currents(const CircuitPiece &piece,
                              RegionAccessor<AccessorType::Generic, ptr_t> fa_in_ptr,
                              RegionAccessor<AccessorType::Generic, ptr_t> fa_out_ptr,
                              RegionAccessor<AccessorType::Generic, PointerLocation> fa_in_loc,
                              RegionAccessor<AccessorType::Generic, PointerLocation> fa_out_loc,
                              RegionAccessor<AccessorType::Generic, float> fa_inductance,
                              RegionAccessor<AccessorType::Generic, float> fa_resistance,
                              RegionAccessor<AccessorType::Generic, float> fa_wire_cap,
                              RegionAccessor<AccessorType::Generic, float> fa_pvt_voltage,
                              RegionAccessor<AccessorType::Generic, float> fa_shr_voltage,
                              RegionAccessor<AccessorType::Generic, float> fa_ghost_voltage,
                              RegionAccessor<AccessorType::Generic, float> *fa_current,
                              RegionAccessor<AccessorType::Generic, float> *fa_voltage)
{
  // See if we can convert all of our accessors to Stuct-of-Array (SOA) accessors
  if (!fa_in_ptr.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_out_ptr.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_in_loc.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_out_loc.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_inductance.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_resistance.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_wire_cap.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_pvt_voltage.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_shr_voltage.can_convert<AccessorType::SOA<0> >()) return false;
  if (!fa_ghost_voltage.can_convert<AccessorType::SOA<0> >()) return false;
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    if (!fa_current[i].can_convert<AccessorType::SOA<0> >()) return false;
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    if (!fa_voltage[i].can_convert<AccessorType::SOA<0> >()) return false;

  RegionAccessor<AccessorType::SOA<sizeof(ptr_t)>,ptr_t> soa_in_ptr = 
    fa_in_ptr.convert<AccessorType::SOA<sizeof(ptr_t)> >();
  RegionAccessor<AccessorType::SOA<sizeof(ptr_t)>,ptr_t> soa_out_ptr = 
    fa_out_ptr.convert<AccessorType::SOA<sizeof(ptr_t)> >();
  RegionAccessor<AccessorType::SOA<sizeof(PointerLocation)>,PointerLocation> soa_in_loc = 
    fa_in_loc.convert<AccessorType::SOA<sizeof(PointerLocation)> >();
  RegionAccessor<AccessorType::SOA<sizeof(PointerLocation)>,PointerLocation> soa_out_loc = 
    fa_out_loc.convert<AccessorType::SOA<sizeof(PointerLocation)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_inductance = 
    fa_inductance.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_resistance = 
    fa_resistance.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_wire_cap = 
    fa_wire_cap.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_pvt_voltage = 
    fa_pvt_voltage.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_shr_voltage = 
    fa_shr_voltage.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_ghost_voltage = 
    fa_ghost_voltage.convert<AccessorType::SOA<sizeof(float)> >();
  // Use malloc here to allocate memory without invoking default constructors
  // which would prematurely test the templated field condition
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> *soa_current = 
    (RegionAccessor<AccessorType::SOA<sizeof(float)>,float>*)malloc(WIRE_SEGMENTS*
        sizeof(RegionAccessor<AccessorType::SOA<sizeof(float)>,float>));
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> *soa_voltage = 
    (RegionAccessor<AccessorType::SOA<sizeof(float)>,float>*)malloc((WIRE_SEGMENTS-1)*
        sizeof(RegionAccessor<AccessorType::SOA<sizeof(float)>,float>));
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    soa_current[i] = fa_current[i].convert<AccessorType::SOA<sizeof(float)> >();
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    soa_voltage[i] = fa_voltage[i].convert<AccessorType::SOA<sizeof(float)> >();

  const int steps = piece.steps;
  unsigned index = 0;
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
      ptr_t current_wire = piece.first_wire+index;
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = _mm512_load_ps(soa_current[i].ptr(current_wire));
        old_i[i] = temp_i[i];
      }
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = _mm512_load_ps(soa_voltage[i].ptr(current_wire));
        old_v[i] = temp_v[i+1];
      }

      // Pin the outer voltages to the node voltages
      temp_v[0] = get_vec_node_voltage(current_wire, soa_pvt_voltage,
                                       soa_shr_voltage, soa_ghost_voltage,
                                       soa_in_ptr, soa_in_loc);
      temp_v[WIRE_SEGMENTS] = get_vec_node_voltage(current_wire, soa_pvt_voltage,
                                       soa_shr_voltage, soa_ghost_voltage,
                                       soa_out_ptr, soa_out_loc);
      __m512 inductance = _mm512_load_ps(soa_inductance.ptr(current_wire));
      __m512 recip_resistance = _mm512_rcp14_ps(_mm512_load_ps(soa_resistance.ptr(current_wire)));
      __m512 recip_capacitance = _mm512_rcp14_ps(_mm512_load_ps(soa_wire_cap.ptr(current_wire)));
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
        _mm512_stream_ps(soa_current[i].ptr(current_wire),temp_i[i]);
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        _mm512_stream_ps(soa_voltage[i].ptr(current_wire),temp_v[i+1]);
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
      ptr_t current_wire = piece.first_wire+index;
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = _mm256_load_ps(soa_current[i].ptr(current_wire));
        old_i[i] = temp_i[i];
      }
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = _mm256_load_ps(soa_voltage[i].ptr(current_wire));
        old_v[i] = temp_v[i+1];
      }

      // Pin the outer voltages to the node voltages
      temp_v[0] = get_vec_node_voltage(current_wire, soa_pvt_voltage,
                                       soa_shr_voltage, soa_ghost_voltage,
                                       soa_in_ptr, soa_in_loc);
      temp_v[WIRE_SEGMENTS] = get_vec_node_voltage(current_wire, soa_pvt_voltage,
                                       soa_shr_voltage, soa_ghost_voltage,
                                       soa_out_ptr, soa_out_loc);
      __m256 inductance = _mm256_load_ps(soa_inductance.ptr(current_wire));
      __m256 recip_resistance = _mm256_rcp_ps(_mm256_load_ps(soa_resistance.ptr(current_wire)));
      __m256 recip_capacitance = _mm256_rcp_ps(_mm256_load_ps(soa_wire_cap.ptr(current_wire)));
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
        _mm256_stream_ps(soa_current[i].ptr(current_wire),temp_i[i]);
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        _mm256_stream_ps(soa_voltage[i].ptr(current_wire),temp_v[i+1]);
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
      ptr_t current_wire = piece.first_wire+index;
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = _mm_load_ps(soa_current[i].ptr(current_wire));
        old_i[i] = temp_i[i];
      }
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = _mm_load_ps(soa_voltage[i].ptr(current_wire));
        old_v[i] = temp_v[i+1];
      }

      // Pin the outer voltages to the node voltages
      temp_v[0] = get_vec_node_voltage(current_wire, soa_pvt_voltage,
                                       soa_shr_voltage, soa_ghost_voltage,
                                       soa_in_ptr, soa_in_loc);
      temp_v[WIRE_SEGMENTS] = get_vec_node_voltage(current_wire, soa_pvt_voltage,
                                       soa_shr_voltage, soa_ghost_voltage,
                                       soa_out_ptr, soa_out_loc);
      __m128 inductance = _mm_load_ps(soa_inductance.ptr(current_wire));
      __m128 recip_resistance = _mm_rcp_ps(_mm_load_ps(soa_resistance.ptr(current_wire)));
      __m128 recip_capacitance = _mm_rcp_ps(_mm_load_ps(soa_wire_cap.ptr(current_wire)));
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
        _mm_stream_ps(soa_current[i].ptr(current_wire),temp_i[i]);
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
        _mm_stream_ps(soa_voltage[i].ptr(current_wire),temp_v[i+1]);
      // Update the index
      index += 4;
    }
  }
#endif
  // Handle any leftover elements (or all of them, in the non-SSE case)
  while (index < piece.num_wires)
  {
    float temp_v[WIRE_SEGMENTS+1];
    float temp_i[WIRE_SEGMENTS];
    float old_i[WIRE_SEGMENTS];
    float old_v[WIRE_SEGMENTS-1];
    ptr_t wire_ptr = piece.first_wire+index;

    for (int i = 0; i < WIRE_SEGMENTS; i++)
    {
      temp_i[i] = soa_current[i].read(wire_ptr);
      old_i[i] = temp_i[i];
    }
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    {
      temp_v[i+1] = soa_voltage[i].read(wire_ptr);
      old_v[i] = temp_v[i+1];
    }

    // Pin the outer voltages to the node voltages
    ptr_t in_ptr = soa_in_ptr.read(wire_ptr);
    PointerLocation in_loc = soa_in_loc.read(wire_ptr);
    temp_v[0] = 
      get_node_voltage(soa_pvt_voltage, soa_shr_voltage, soa_ghost_voltage, in_loc, in_ptr);
    ptr_t out_ptr = soa_out_ptr.read(wire_ptr);
    PointerLocation out_loc = soa_out_loc.read(wire_ptr);
    temp_v[WIRE_SEGMENTS] = 
      get_node_voltage(soa_pvt_voltage, soa_shr_voltage, soa_ghost_voltage, out_loc, out_ptr);

    // Solve the RLC model iteratively
    float inductance = soa_inductance.read(wire_ptr);
    float recip_resistance = 1.f/soa_resistance.read(wire_ptr);
    float recip_capacitance = 1.f/soa_wire_cap.read(wire_ptr);
    float dt = piece.dt;
    float recip_dt = 1.0/dt;
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
      soa_current[i].write(wire_ptr, temp_i[i]);
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      soa_voltage[i].write(wire_ptr, temp_v[i+1]);
    // Update the index
    index++;
  }
  // Clean up
  free(soa_current);
  free(soa_voltage);
  return true;
}

/*static*/
void CalcNewCurrentsTask::cpu_base_impl(const CircuitPiece &p,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context ctx, HighLevelRuntime* rt)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::Generic, float> fa_current[WIRE_SEGMENTS];
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    fa_current[i] = regions[0].get_field_accessor(FID_CURRENT+i).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_voltage[WIRE_SEGMENTS-1];
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    fa_voltage[i] = regions[0].get_field_accessor(FID_WIRE_VOLTAGE+i).typeify<float>();
  RegionAccessor<AccessorType::Generic, ptr_t> fa_in_ptr = 
    regions[1].get_field_accessor(FID_IN_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, ptr_t> fa_out_ptr = 
    regions[1].get_field_accessor(FID_OUT_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_in_loc = 
    regions[1].get_field_accessor(FID_IN_LOC).typeify<PointerLocation>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_out_loc = 
    regions[1].get_field_accessor(FID_OUT_LOC).typeify<PointerLocation>();
  RegionAccessor<AccessorType::Generic, float> fa_inductance = 
    regions[1].get_field_accessor(FID_INDUCTANCE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_resistance = 
    regions[1].get_field_accessor(FID_RESISTANCE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_wire_cap = 
    regions[1].get_field_accessor(FID_WIRE_CAP).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_pvt_voltage = 
    regions[2].get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_voltage =
    regions[3].get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_ghost_voltage = 
    regions[4].get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();

  // See if we can do the dense version with vector instructions
  if (dense_calc_new_currents(p, fa_in_ptr, fa_out_ptr, fa_in_loc, fa_out_loc,
                              fa_inductance, fa_resistance, fa_wire_cap,
                              fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage,
                              fa_current, fa_voltage))
    return;

  LegionRuntime::HighLevel::IndexIterator itr(rt, ctx, p.pvt_wires);
  float temp_v[WIRE_SEGMENTS+1];
  float temp_i[WIRE_SEGMENTS];
  float old_i[WIRE_SEGMENTS];
  float old_v[WIRE_SEGMENTS-1];
  while (itr.has_next())
  {
    ptr_t wire_ptr = itr.next();
    const float dt = p.dt;
    const float recip_dt = 1.0f / dt;
    const int steps = p.steps;

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
    ptr_t in_ptr = fa_in_ptr.read(wire_ptr);
    PointerLocation in_loc = fa_in_loc.read(wire_ptr);
    temp_v[0] = 
      get_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, in_loc, in_ptr);
    ptr_t out_ptr = fa_out_ptr.read(wire_ptr);
    PointerLocation out_loc = fa_out_loc.read(wire_ptr);
    temp_v[WIRE_SEGMENTS] = 
      get_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, out_loc, out_ptr);

    // Solve the RLC model iteratively
    float inductance = fa_inductance.read(wire_ptr);
    float recip_resistance = 1.0f / fa_resistance.read(wire_ptr);
    float recip_capacitance = 1.0f / fa_wire_cap.read(wire_ptr);
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
                                         const std::vector<PhysicalRegion> &regions,
                                         Context ctx, HighLevelRuntime* rt)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::Generic, ptr_t> fa_in_ptr = 
    regions[0].get_field_accessor(FID_IN_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, ptr_t> fa_out_ptr = 
    regions[0].get_field_accessor(FID_OUT_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_in_loc = 
    regions[0].get_field_accessor(FID_IN_LOC).typeify<PointerLocation>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_out_loc = 
    regions[0].get_field_accessor(FID_OUT_LOC).typeify<PointerLocation>();
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

  LegionRuntime::HighLevel::IndexIterator itr(rt, ctx, p.pvt_wires);
  while (itr.has_next())
  {
    ptr_t wire_ptr = itr.next();
#ifdef DEBUG_MATH
    printf("DC: %d = %f->(%d,%d), %f->(%d,%d)\n",
           wire_ptr.value,
           fa_in_current.read(wire_ptr),
           fa_in_ptr.read(wire_ptr).value, fa_in_loc.read(wire_ptr),
           fa_out_current.read(wire_ptr),
           fa_out_ptr.read(wire_ptr).value, fa_out_loc.read(wire_ptr));
#endif
    const float dt = p.dt;
    float in_current = -dt * fa_in_current.read(wire_ptr);
    float out_current = dt * fa_out_current.read(wire_ptr);
    ptr_t in_ptr = fa_in_ptr.read(wire_ptr);
    ptr_t out_ptr = fa_out_ptr.read(wire_ptr);
    PointerLocation in_loc = fa_in_loc.read(wire_ptr);
    PointerLocation out_loc = fa_out_loc.read(wire_ptr);

    reduce_node<AccumulateCharge>(fa_pvt_charge, fa_shr_charge, fa_ghost_charge,
                                  in_loc, in_ptr, in_current);
    reduce_node<AccumulateCharge>(fa_pvt_charge, fa_shr_charge, fa_ghost_charge,
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
                                   const RegionAccessor<AT,float> &fa_cap,
                                   const RegionAccessor<AT,float> &fa_leakage,
                                   Context ctx, HighLevelRuntime* rt)
{
  IndexIterator itr(rt, ctx, lr);
  while (itr.has_next())
  {
    ptr_t node_ptr = itr.next();
    float voltage = fa_voltage.read(node_ptr);
    float charge = fa_charge.read(node_ptr);
    float capacitance = fa_cap.read(node_ptr);
    float leakage = fa_leakage.read(node_ptr);
    voltage += charge / capacitance;
    voltage *= (1.f - leakage);
    fa_voltage.write(node_ptr, voltage);
    // Reset the charge for the next iteration
    fa_charge.write(node_ptr, 0.f);
  }
}

/*static*/
void UpdateVoltagesTask::cpu_base_impl(const CircuitPiece &p,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, HighLevelRuntime* rt)
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
  RegionAccessor<AccessorType::Generic, float> fa_pvt_cap = 
    regions[2].get_field_accessor(FID_NODE_CAP).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_pvt_leakage = 
    regions[2].get_field_accessor(FID_LEAKAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_cap = 
    regions[3].get_field_accessor(FID_NODE_CAP).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_leakage = 
    regions[3].get_field_accessor(FID_LEAKAGE).typeify<float>();
  // Don't need this for the CPU version
  //RegionAccessor<AccessorType::Generic, PointerLocation> fa_location = 
  //  regions[4].get_field_accessor(FID_LOCATOR).typeify<PointerLocation>();

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
  IndexIterator itr(runtime, ctx, lr);
  bool success = true;
  while (itr.has_next() && success)
  {
    ptr_t ptr = itr.next();
    float value = fa_check.read(ptr);
    if (std::isnan(value))
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

