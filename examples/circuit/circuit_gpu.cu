/* Copyright 2016 Stanford University
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

#include "cuda_runtime.h"

using namespace LegionRuntime::Accessor;

class GPUAccumulateCharge {
public:
  typedef float LHS;
  typedef float RHS;

  template<bool EXCLUSIVE>
  __device__ __forceinline__
  static void apply(LHS &lhs, RHS &rhs)
  {
    float *target = &lhs; 
    atomicAdd(target,rhs);
  }

  template<bool EXCLUSIVE>
  __device__ __forceinline__
  static void fold(RHS &rhs1, RHS rhs2)
  {
    float *target = &rhs1;
    atomicAdd(target,rhs2);
  }
};

template<typename AT, int SEGMENTS>
struct SegmentAccessors {
public:
  SegmentAccessors(AT *fas)
  {
    for (int i = 0; i < SEGMENTS; i++)
      accessors[i] = fas[i];
  }
public:
  AT accessors[SEGMENTS];
};

template<typename AT>
__device__ __forceinline__
float find_node_voltage(const RegionAccessor<AT,float> &pvt,
                        const RegionAccessor<AT,float> &shr,
                        const RegionAccessor<AT,float> &ghost,
                        ptr_t ptr, PointerLocation loc)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      return pvt.read(ptr);
    case SHARED_PTR:
      return shr.read(ptr);
    case GHOST_PTR:
      return ghost.read(ptr);
    default:
      break; // assert(false);
  }
  return 0.f;
}

template<typename AT1, typename AT2>
__global__
void calc_new_currents_kernel(ptr_t first,
                              int num_wires,
			      float dt,
			      int steps,
                              RegionAccessor<AT1,ptr_t> fa_in_ptr,
                              RegionAccessor<AT1,ptr_t> fa_out_ptr,
                              RegionAccessor<AT2,PointerLocation> fa_in_loc,
                              RegionAccessor<AT2,PointerLocation> fa_out_loc,
                              RegionAccessor<AT2,float> fa_inductance,
                              RegionAccessor<AT2,float> fa_resistance,
                              RegionAccessor<AT2,float> fa_wire_cap,
                              RegionAccessor<AT2,float> fa_pvt_voltage,
                              RegionAccessor<AT2,float> fa_shr_voltage,
                              RegionAccessor<AT2,float> fa_ghost_voltage,
                              SegmentAccessors<RegionAccessor<AT2,float>,WIRE_SEGMENTS> fa_currents,
                              SegmentAccessors<RegionAccessor<AT2,float>,WIRE_SEGMENTS-1> fa_voltages)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // We can do this because we know we have SOA layout and wires are dense
  if (tid < num_wires)
  {
    ptr_t wire_ptr = first + tid;
    float recip_dt = 1.f/dt;

    float temp_v[WIRE_SEGMENTS+1];
    float temp_i[WIRE_SEGMENTS];
    float old_i[WIRE_SEGMENTS];
    float old_v[WIRE_SEGMENTS-1];

    #pragma unroll
    for (int i = 0; i < WIRE_SEGMENTS; i++)
    {
      temp_i[i] = fa_currents.accessors[i].read(wire_ptr);
      old_i[i] = temp_i[i];
    }
    #pragma unroll
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    {
      temp_v[i+1] = fa_voltages.accessors[i].read(wire_ptr);
      old_v[i] = temp_v[i+1];
    }

    ptr_t in_ptr = fa_in_ptr.read(wire_ptr);
    PointerLocation in_loc = fa_in_loc.read(wire_ptr);
    temp_v[0] = 
      find_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, in_ptr, in_loc);
    ptr_t out_ptr = fa_out_ptr.read(wire_ptr);
    PointerLocation out_loc = fa_out_loc.read(wire_ptr);
    temp_v[WIRE_SEGMENTS] = 
      find_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, in_ptr, in_loc);

    // Solve the RLC model iteratively
    float inductance = fa_inductance.read(wire_ptr);
    float recip_resistance = 1.f/fa_resistance.read(wire_ptr);
    float recip_capacitance = 1.f/fa_wire_cap.read(wire_ptr);
    for (int j = 0; j < steps; j++)
    {
      #pragma unroll
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        temp_i[i] = ((temp_v[i] - temp_v[i+1]) -
                     (inductance * (temp_i[i] - old_i[i]) * recip_dt)) * recip_resistance;
      }
      #pragma unroll
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        temp_v[i+1] = old_v[i] + dt * (temp_i[i] - temp_i[i+1]) * recip_capacitance;
      }
    }

    // Write out the result
    #pragma unroll
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      fa_currents.accessors[i].write(wire_ptr, temp_i[i]);
    #pragma unroll
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      fa_voltages.accessors[i].write(wire_ptr, temp_v[i+1]);
  }
}

/*static*/
__host__
void CalcNewCurrentsTask::gpu_base_impl(const CircuitPiece &piece,
                                        const std::vector<PhysicalRegion> &regions)
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
  // We better be able to convert all of our accessors to SOA for the GPU
  // If no we'll assert.
  if (!fa_in_ptr.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_out_ptr.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_in_loc.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_out_loc.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_inductance.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_resistance.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_wire_cap.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_pvt_voltage.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_shr_voltage.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_ghost_voltage.can_convert<AccessorType::SOA<0> >()) assert(false);
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    if (!fa_current[i].can_convert<AccessorType::SOA<0> >()) assert(false);
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    if (!fa_voltage[i].can_convert<AccessorType::SOA<0> >()) assert(false);

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

  const int threads_per_block = 256;
  const int num_blocks = (piece.num_wires + (threads_per_block-1)) / threads_per_block;

  // Create compound types so CUDA copies them by value
  SegmentAccessors<RegionAccessor<AccessorType::SOA<sizeof(float)>,float>,WIRE_SEGMENTS>
    current_segments(soa_current);
  SegmentAccessors<RegionAccessor<AccessorType::SOA<sizeof(float)>,float>,(WIRE_SEGMENTS-1)>
    voltage_segments(soa_voltage);

  calc_new_currents_kernel<<<num_blocks,threads_per_block>>>(piece.first_wire,
                                                             piece.num_wires,
                                                             piece.dt,
                                                             piece.steps,
                                                             soa_in_ptr,
                                                             soa_out_ptr,
                                                             soa_in_loc,
                                                             soa_out_loc,
                                                             soa_inductance,
                                                             soa_resistance,
                                                             soa_wire_cap,
                                                             soa_pvt_voltage,
                                                             soa_shr_voltage,
                                                             soa_ghost_voltage,
                                                             current_segments,
                                                             voltage_segments);

  free(soa_current);
  free(soa_voltage);
#endif
}

template<typename REDOP, typename AT1, typename AT2>
__device__ __forceinline__
void reduce_local(const RegionAccessor<AT1, float> &pvt,
                  const RegionAccessor<AT2, float> &shr,
                  const RegionAccessor<AT2, float> &ghost,
                  ptr_t ptr, PointerLocation loc, typename REDOP::RHS value)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      pvt.template reduce<REDOP>(ptr, value);
      break;
    case SHARED_PTR:
      shr.reduce(ptr, value);
      break;
    case GHOST_PTR:
      ghost.reduce(ptr, value);
      break;
    default:
      break; // assert(false); // should never make it here
  }
}

template<typename AT1, typename AT2, typename AT3>
__global__
void distribute_charge_kernel(ptr_t first,
                              const int num_wires,
			      float dt,
                              RegionAccessor<AT2,ptr_t> fa_in_ptr,
                              RegionAccessor<AT2,ptr_t> fa_out_ptr,
                              RegionAccessor<AT1,PointerLocation> fa_in_loc,
                              RegionAccessor<AT1,PointerLocation> fa_out_loc,
                              RegionAccessor<AT1,float> fa_in_current,
                              RegionAccessor<AT1,float> fa_out_current,
                              RegionAccessor<AT1,float> fa_pvt_charge,
                              RegionAccessor<AT3,float> fa_shr_charge,
                              RegionAccessor<AT3,float> fa_ghost_charge)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid < num_wires)
  {
    ptr_t wire_ptr = first + tid;

    float in_dq = -dt * fa_in_current.read(wire_ptr);
    float out_dq = dt * fa_out_current.read(wire_ptr);
    
    ptr_t in_ptr = fa_in_ptr.read(wire_ptr);
    PointerLocation in_loc = fa_in_loc.read(wire_ptr);
    reduce_local<GPUAccumulateCharge>(fa_pvt_charge, fa_shr_charge, fa_ghost_charge,
                                      in_ptr, in_loc, in_dq);

    ptr_t out_ptr = fa_out_ptr.read(wire_ptr);
    PointerLocation out_loc = fa_out_loc.read(wire_ptr);
    reduce_local<GPUAccumulateCharge>(fa_pvt_charge, fa_shr_charge, fa_ghost_charge,
                                      out_ptr, out_loc, out_dq);
  }
}

/*static*/
__host__
void DistributeChargeTask::gpu_base_impl(const CircuitPiece &piece,
                                         const std::vector<PhysicalRegion> &regions)
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
  RegionAccessor<AccessorType::Generic, float> fa_shr_charge = 
    regions[2].get_accessor().typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_ghost_charge =
    regions[3].get_accessor().typeify<float>();

  if (!fa_in_ptr.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_out_ptr.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_in_loc.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_out_loc.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_in_current.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_out_current.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_pvt_charge.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_shr_charge.can_convert<AccessorType::ReductionFold<GPUAccumulateCharge> >()) assert(false);
  if (!fa_ghost_charge.can_convert<AccessorType::ReductionFold<GPUAccumulateCharge> >()) assert(false);

  RegionAccessor<AccessorType::SOA<sizeof(ptr_t)>,ptr_t> soa_in_ptr = 
    fa_in_ptr.convert<AccessorType::SOA<sizeof(ptr_t)> >();
  RegionAccessor<AccessorType::SOA<sizeof(ptr_t)>,ptr_t> soa_out_ptr = 
    fa_out_ptr.convert<AccessorType::SOA<sizeof(ptr_t)> >();
  RegionAccessor<AccessorType::SOA<sizeof(PointerLocation)>,PointerLocation> soa_in_loc = 
    fa_in_loc.convert<AccessorType::SOA<sizeof(PointerLocation)> >();
  RegionAccessor<AccessorType::SOA<sizeof(PointerLocation)>,PointerLocation> soa_out_loc = 
    fa_out_loc.convert<AccessorType::SOA<sizeof(PointerLocation)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_in_current = 
    fa_in_current.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_out_current = 
    fa_out_current.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_pvt_charge = 
    fa_pvt_charge.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::ReductionFold<GPUAccumulateCharge>,float> fold_shr_charge = 
    fa_shr_charge.convert<AccessorType::ReductionFold<GPUAccumulateCharge> >();
  RegionAccessor<AccessorType::ReductionFold<GPUAccumulateCharge>,float> fold_ghost_charge = 
    fa_ghost_charge.convert<AccessorType::ReductionFold<GPUAccumulateCharge> >();

  const int threads_per_block = 256;
  const int num_blocks = (piece.num_wires + (threads_per_block-1)) / threads_per_block;

  distribute_charge_kernel<<<num_blocks,threads_per_block>>>(piece.first_wire,
                                                             piece.num_wires,
                                                             piece.dt,
                                                             soa_in_ptr,
                                                             soa_out_ptr,
                                                             soa_in_loc,
                                                             soa_out_loc,
                                                             soa_in_current,
                                                             soa_out_current,
                                                             soa_pvt_charge,
                                                             fold_shr_charge,
                                                             fold_ghost_charge);
#endif
}

template<typename AT>
__global__
void update_voltages_kernel(ptr_t first,
                            const int num_nodes,
                            RegionAccessor<AT,float> fa_pvt_voltage,
                            RegionAccessor<AT,float> fa_shr_voltage,
                            RegionAccessor<AT,float> fa_pvt_charge,
                            RegionAccessor<AT,float> fa_shr_charge,
                            RegionAccessor<AT,float> fa_pvt_cap,
                            RegionAccessor<AT,float> fa_shr_cap,
                            RegionAccessor<AT,float> fa_pvt_leakage,
                            RegionAccessor<AT,float> fa_shr_leakage,
                            RegionAccessor<AT,PointerLocation> fa_ptr_loc)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_nodes)
  {
    ptr_t node_ptr = first + tid;
    PointerLocation node_loc = fa_ptr_loc.read(node_ptr);
    if (node_loc == PRIVATE_PTR)
    {
      float voltage = fa_pvt_voltage.read(node_ptr);
      float charge = fa_pvt_charge.read(node_ptr);
      float capacitance = fa_pvt_cap.read(node_ptr);
      float leakage = fa_pvt_leakage.read(node_ptr);
      voltage += (charge / capacitance);
      voltage *= (1.f - leakage);
      fa_pvt_voltage.write(node_ptr, voltage);
      fa_pvt_charge.write(node_ptr, 0.f);
    }
    else
    {
      float voltage = fa_shr_voltage.read(node_ptr);
      float charge = fa_shr_charge.read(node_ptr);
      float capacitance = fa_shr_cap.read(node_ptr);
      float leakage = fa_shr_leakage.read(node_ptr);
      voltage += (charge / capacitance);
      voltage *= (1.f - leakage);
      fa_pvt_voltage.write(node_ptr, voltage);
      fa_pvt_charge.write(node_ptr, 0.f);
    }
  }
}

/*static*/
__host__
void UpdateVoltagesTask::gpu_base_impl(const CircuitPiece &piece,
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
  RegionAccessor<AccessorType::Generic, float> fa_pvt_cap = 
    regions[2].get_field_accessor(FID_NODE_CAP).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_pvt_leakage = 
    regions[2].get_field_accessor(FID_LEAKAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_cap = 
    regions[3].get_field_accessor(FID_NODE_CAP).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_shr_leakage = 
    regions[3].get_field_accessor(FID_LEAKAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_location = 
    regions[4].get_field_accessor(FID_LOCATOR).typeify<PointerLocation>();

  if (!fa_pvt_voltage.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_pvt_charge.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_shr_voltage.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_shr_charge.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_pvt_cap.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_pvt_leakage.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_shr_cap.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_shr_leakage.can_convert<AccessorType::SOA<0> >()) assert(false);
  if (!fa_location.can_convert<AccessorType::SOA<0> >()) assert(false);

  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_pvt_voltage = 
    fa_pvt_voltage.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_pvt_charge = 
    fa_pvt_charge.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_shr_voltage = 
    fa_shr_voltage.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_shr_charge = 
    fa_shr_charge.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_pvt_cap = 
    fa_pvt_cap.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_pvt_leakage = 
    fa_pvt_leakage.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_shr_cap = 
    fa_shr_cap.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>,float> soa_shr_leakage = 
    fa_shr_leakage.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(PointerLocation)>,PointerLocation> soa_ptr_loc = 
    fa_location.convert<AccessorType::SOA<sizeof(PointerLocation)> >();

  const int threads_per_block = 256;
  const int num_blocks = (piece.num_nodes + (threads_per_block-1)) / threads_per_block;

  update_voltages_kernel<<<num_blocks,threads_per_block>>>(piece.first_node,
                                                           piece.num_nodes,
                                                           soa_pvt_voltage,
                                                           soa_shr_voltage,
                                                           soa_pvt_charge,
                                                           soa_shr_charge,
                                                           soa_pvt_cap,
                                                           soa_shr_cap,
                                                           soa_pvt_leakage,
                                                           soa_shr_leakage,
                                                           soa_ptr_loc);
#endif
}

