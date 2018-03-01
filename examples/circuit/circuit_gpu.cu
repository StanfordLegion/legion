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

#include "cuda_runtime.h"

class GPUAccumulateCharge {
public:
  typedef float LHS;
  typedef float RHS;

  template<bool EXCLUSIVE>
  __host__ __device__ __forceinline__
  static void apply(LHS &lhs, RHS &rhs)
  {
#ifdef __CUDA_ARCH__
    float *target = &lhs; 
    atomicAdd(target,rhs);
#else
    assert(false);
#endif
  }

  template<bool EXCLUSIVE>
  __host__ __device__ __forceinline__
  static void fold(RHS &rhs1, RHS rhs2)
  {
#ifdef __CUDA_ARCH__
    float *target = &rhs1;
    atomicAdd(target,rhs2);
#else
    assert(false);
#endif
  }
};

template<typename AT, int SEGMENTS>
struct SegmentAccessors {
public:
  __host__ __device__
  inline AT& operator[](unsigned index) { return accessors[index]; }
  __host__ __device__
  inline const AT& operator[](unsigned index) const { return accessors[index]; }
public:
  AT accessors[SEGMENTS];
};

__device__ __forceinline__
float find_node_voltage(const AccessorROfloat &pvt,
                        const AccessorROfloat &shr,
                        const AccessorROfloat &ghost,
                        Point<1> ptr, PointerLocation loc)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      return pvt[ptr];
    case SHARED_PTR:
      return shr[ptr];
    case GHOST_PTR:
      return ghost[ptr];
    default:
      break; // assert(false);
  }
  return 0.f;
}

__global__
void calc_new_currents_kernel(Point<1> first,
                              int num_wires,
			      float dt,
			      int steps,
                              const AccessorROpoint fa_in_ptr,
                              const AccessorROpoint fa_out_ptr,
                              const AccessorROloc fa_in_loc,
                              const AccessorROloc fa_out_loc,
                              const AccessorROfloat fa_inductance,
                              const AccessorROfloat fa_resistance,
                              const AccessorROfloat fa_wire_cap,
                              const AccessorROfloat fa_pvt_voltage,
                              const AccessorROfloat fa_shr_voltage,
                              const AccessorROfloat fa_ghost_voltage,
                              const SegmentAccessors<AccessorRWfloat,WIRE_SEGMENTS> fa_currents,
                              const SegmentAccessors<AccessorRWfloat,WIRE_SEGMENTS-1> fa_voltages)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // We can do this because we know we have SOA layout and wires are dense
  if (tid < num_wires)
  {
    const Point<1> wire_ptr = first + tid;
    float recip_dt = 1.f/dt;

    float temp_v[WIRE_SEGMENTS+1];
    float temp_i[WIRE_SEGMENTS];
    float old_i[WIRE_SEGMENTS];
    float old_v[WIRE_SEGMENTS-1];

    #pragma unroll
    for (int i = 0; i < WIRE_SEGMENTS; i++)
    {
      temp_i[i] = fa_currents[i][wire_ptr];
      old_i[i] = temp_i[i];
    }
    #pragma unroll
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    {
      temp_v[i+1] = fa_voltages[i][wire_ptr];
      old_v[i] = temp_v[i+1];
    }

    Point<1> in_ptr = fa_in_ptr[wire_ptr];
    PointerLocation in_loc = fa_in_loc[wire_ptr];
    temp_v[0] = 
      find_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, in_ptr, in_loc);
    Point<1> out_ptr = fa_out_ptr[wire_ptr];
    PointerLocation out_loc = fa_out_loc[wire_ptr];
    temp_v[WIRE_SEGMENTS] = 
      find_node_voltage(fa_pvt_voltage, fa_shr_voltage, fa_ghost_voltage, in_ptr, in_loc);

    // Solve the RLC model iteratively
    float inductance = fa_inductance[wire_ptr];
    float recip_resistance = 1.f/fa_resistance[wire_ptr];
    float recip_capacitance = 1.f/fa_wire_cap[wire_ptr];
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
      fa_currents[i][wire_ptr] = temp_i[i];
    #pragma unroll
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      fa_voltages[i][wire_ptr] = temp_v[i+1];
  }
}

/*static*/
__host__
void CalcNewCurrentsTask::gpu_base_impl(const CircuitPiece &piece,
                                        const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  SegmentAccessors<AccessorRWfloat,WIRE_SEGMENTS> fa_currents;
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    fa_currents[i] = AccessorRWfloat(regions[0], FID_CURRENT+i);
  SegmentAccessors<AccessorRWfloat,WIRE_SEGMENTS-1> fa_voltages;
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    fa_voltages[i] = AccessorRWfloat(regions[0], FID_WIRE_VOLTAGE+i);

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

  const int threads_per_block = 256;
  const int num_blocks = (piece.num_wires + (threads_per_block-1)) / threads_per_block;

  calc_new_currents_kernel<<<num_blocks,threads_per_block>>>(piece.first_wire,
                                                             piece.num_wires,
                                                             piece.dt,
                                                             piece.steps,
                                                             fa_in_ptr,
                                                             fa_out_ptr,
                                                             fa_in_loc,
                                                             fa_out_loc,
                                                             fa_inductance,
                                                             fa_resistance,
                                                             fa_wire_cap,
                                                             fa_pvt_voltage,
                                                             fa_shr_voltage,
                                                             fa_ghost_voltage,
                                                             fa_currents,
                                                             fa_voltages);
#endif
}

typedef ReductionAccessor<GPUAccumulateCharge,false/*exclusive*/,1,coord_t,
                          Realm::AffineAccessor<float,1,coord_t> > AccessorRDfloat;

__device__ __forceinline__
void reduce_local(const AccessorRWfloat &pvt,
                  const AccessorRDfloat &shr,
                  const AccessorRDfloat &ghost,
                  Point<1> ptr, PointerLocation loc, float value)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      GPUAccumulateCharge::apply<true/*exclusive*/>(pvt[ptr], value);
      break;
    case SHARED_PTR:
      shr[ptr] <<= value;
      break;
    case GHOST_PTR:
      ghost[ptr] <<= value;
      break;
    default:
      break; // assert(false); // should never make it here
  }
}

__global__
void distribute_charge_kernel(Point<1> first,
                              const int num_wires,
			      float dt,
                              const AccessorROpoint fa_in_ptr,
                              const AccessorROpoint fa_out_ptr,
                              const AccessorROloc fa_in_loc,
                              const AccessorROloc fa_out_loc,
                              const AccessorROfloat fa_in_current,
                              const AccessorROfloat fa_out_current,
                              const AccessorRWfloat fa_pvt_charge,
                              const AccessorRDfloat fa_shr_charge,
                              const AccessorRDfloat fa_ghost_charge)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid < num_wires)
  {
    const Point<1> wire_ptr = first + tid;

    float in_dq = -dt * fa_in_current[wire_ptr];
    float out_dq = dt * fa_out_current[wire_ptr];
    
    Point<1> in_ptr = fa_in_ptr[wire_ptr];
    PointerLocation in_loc = fa_in_loc[wire_ptr];
    reduce_local(fa_pvt_charge, fa_shr_charge, fa_ghost_charge, in_ptr, in_loc, in_dq);

    Point<1> out_ptr = fa_out_ptr[wire_ptr];
    PointerLocation out_loc = fa_out_loc[wire_ptr];
    reduce_local(fa_pvt_charge, fa_shr_charge, fa_ghost_charge, out_ptr, out_loc, out_dq);
  }
}

/*static*/
__host__
void DistributeChargeTask::gpu_base_impl(const CircuitPiece &piece,
                                         const std::vector<PhysicalRegion> &regions)
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

  const int threads_per_block = 256;
  const int num_blocks = (piece.num_wires + (threads_per_block-1)) / threads_per_block;

  distribute_charge_kernel<<<num_blocks,threads_per_block>>>(piece.first_wire,
                                                             piece.num_wires,
                                                             piece.dt,
                                                             fa_in_ptr,
                                                             fa_out_ptr,
                                                             fa_in_loc,
                                                             fa_out_loc,
                                                             fa_in_current,
                                                             fa_out_current,
                                                             fa_pvt_charge,
                                                             fa_shr_charge,
                                                             fa_ghost_charge);
#endif
}

__global__
void update_voltages_kernel(Point<1> first,
                            const int num_nodes,
                            const AccessorRWfloat fa_pvt_voltage,
                            const AccessorRWfloat fa_shr_voltage,
                            const AccessorRWfloat fa_pvt_charge,
                            const AccessorRWfloat fa_shr_charge,
                            const AccessorROfloat fa_pvt_cap,
                            const AccessorROfloat fa_shr_cap,
                            const AccessorROfloat fa_pvt_leakage,
                            const AccessorROfloat fa_shr_leakage,
                            const AccessorROloc fa_ptr_loc)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_nodes)
  {
    const Point<1> node_ptr = first + tid;
    PointerLocation node_loc = fa_ptr_loc[node_ptr];
    if (node_loc == PRIVATE_PTR)
    {
      float voltage = fa_pvt_voltage[node_ptr];
      float charge = fa_pvt_charge[node_ptr];
      float capacitance = fa_pvt_cap[node_ptr];
      float leakage = fa_pvt_leakage[node_ptr];
      voltage += (charge / capacitance);
      voltage *= (1.f - leakage);
      fa_pvt_voltage[node_ptr] = voltage;
      fa_pvt_charge[node_ptr] = 0.f;
    }
    else
    {
      float voltage = fa_shr_voltage[node_ptr];
      float charge = fa_shr_charge[node_ptr];
      float capacitance = fa_shr_cap[node_ptr];
      float leakage = fa_shr_leakage[node_ptr];
      voltage += (charge / capacitance);
      voltage *= (1.f - leakage);
      fa_pvt_voltage[node_ptr] = voltage;
      fa_pvt_charge[node_ptr] = 0.f;
    }
  }
}

/*static*/
__host__
void UpdateVoltagesTask::gpu_base_impl(const CircuitPiece &piece,
                                       const std::vector<PhysicalRegion> &regions)
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

  const AccessorROloc fa_ptr_loc(regions[4], FID_LOCATOR);

  const int threads_per_block = 256;
  const int num_blocks = (piece.num_nodes + (threads_per_block-1)) / threads_per_block;

  update_voltages_kernel<<<num_blocks,threads_per_block>>>(piece.first_node,
                                                           piece.num_nodes,
                                                           fa_pvt_voltage,
                                                           fa_shr_voltage,
                                                           fa_pvt_charge,
                                                           fa_shr_charge,
                                                           fa_pvt_cap,
                                                           fa_shr_cap,
                                                           fa_pvt_leakage,
                                                           fa_shr_leakage,
                                                           fa_ptr_loc);
#endif
}

