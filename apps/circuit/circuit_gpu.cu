/* Copyright 2014 Stanford University
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

#include "cuda.h"
#include "cuda_runtime.h"

#define CUDA_SAFE_CALL(expr)				\
	{						\
		cudaError_t err = (expr);		\
		if (err != cudaSuccess)			\
		{					\
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			assert(false);			\
		}					\
	}

using namespace LegionRuntime::Accessor;

class GPUAccumulateCharge {
public:
  typedef CircuitNode LHS;
  typedef float RHS;

  template<bool EXCLUSIVE>
  __device__ __forceinline__
  static void apply(LHS &lhs, RHS &rhs)
  {
    float *target = &(lhs.charge); 
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

// Helper methods

template<typename AT>
__device__ __forceinline__
CircuitNode& get_node(const RegionAccessor<AT, CircuitNode> &pvt,
                      const RegionAccessor<AT, CircuitNode> &owned,
                      const RegionAccessor<AT, CircuitNode> &ghost,
                      PointerLocation loc, ptr_t ptr)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      //assert((pvt.first_elmt <= ptr.value) && (ptr.value <= pvt.last_elmt));
      return pvt.ref(ptr);
    case SHARED_PTR:
      //assert((owned.first_elmt <= ptr.value) && (ptr.value <= owned.last_elmt));
      return owned.ref(ptr);
    case GHOST_PTR:
      //assert((ghost.first_elmt <= ptr.value) && (ptr.value <= ghost.last_elmt));
      return ghost.ref(ptr);
    default:
      assert(false);
  }
  return pvt.ref(ptr);
}

template<typename REDOP, typename AT1, typename AT2>
__device__ __forceinline__
void reduce_local(const RegionAccessor<AT1, CircuitNode> &pvt,
                  const RegionAccessor<AT2, CircuitNode> &owned,
                  const RegionAccessor<AT2, CircuitNode> &ghost,
                  PointerLocation loc, ptr_t ptr, typename REDOP::RHS value)
{
  switch (loc)
  {
    case PRIVATE_PTR:
      pvt.template reduce<REDOP>(ptr, value);
      break;
    case SHARED_PTR:
      owned.reduce(ptr, value);
      break;
    case GHOST_PTR:
      ghost.reduce(ptr, value);
      break;
    default:
      assert(false);
  }
}

// Actual kernels

template<typename AT>
__global__
void calc_new_currents_kernel(ptr_t first,
                              int num_wires,
			      float dt,
			      int steps,
                              RegionAccessor<AT,CircuitWire> wires,
                              RegionAccessor<AT,CircuitNode> pvt,
                              RegionAccessor<AT,CircuitNode> owned,
                              RegionAccessor<AT,CircuitNode> ghost)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 

  if (tid < num_wires)
  {
    ptr_t local_ptr = first + tid;
    CircuitWire &wire = wires.ref(local_ptr);
    CircuitNode &in_node = get_node(pvt, owned, ghost, wire.in_loc, wire.in_ptr);
    CircuitNode &out_node = get_node(pvt, owned, ghost, wire.out_loc, wire.out_ptr);

    // Solve RLC model iteratively
    float recip_dt = 1.f/dt;
    float new_i[WIRE_SEGMENTS];
    float new_v[WIRE_SEGMENTS+1];
    #pragma unroll
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      new_i[i] = wire.current[i];
    #pragma unroll
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      new_v[i+1] = wire.voltage[i];
    new_v[0] = in_node.voltage;
    new_v[WIRE_SEGMENTS] = out_node.voltage;

    float recip_resistance = 1.f/wire.resistance;
    float recip_capacitance = 1.f/wire.capacitance;

    for (int j = 0; j < steps; j++)
    {
      // first, figure out the new current from the voltage differential
      // and our inductance:
      // dV = R*I + L*I' ==> I = (dV - L*I')/R
      #pragma unroll
      for (int i = 0; i < WIRE_SEGMENTS; i++)
      {
        new_i[i] = ((new_v[i] - new_v[i+1]) - 
                    (wire.inductance*(new_i[i] - wire.current[i]) * recip_dt)) * recip_resistance;
      }
      // Now update the inter-node voltages
      #pragma unroll
      for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      {
        new_v[i+1] = wire.voltage[i] + dt*(new_i[i] - new_i[i+1]) * recip_capacitance;
      }
    }

    // Copy everything back
    #pragma unroll
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      wire.current[i] = new_i[i];
    #pragma unroll
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      wire.voltage[i] = new_v[i+1];
  }
}

__host__
void calc_new_currents_gpu(CircuitPiece *p,
                           const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::SOA<0>, CircuitWire> wires = 
    extract_accessor<AccessorType::SOA<0>,CircuitWire>(regions[0]);
  RegionAccessor<AccessorType::SOA<0>, CircuitNode> pvt = 
    extract_accessor<AccessorType::SOA<0>,CircuitNode>(regions[1]);
  RegionAccessor<AccessorType::SOA<0>, CircuitNode> owned = 
    extract_accessor<AccessorType::SOA<0>,CircuitNode>(regions[2]);
  RegionAccessor<AccessorType::SOA<0>, CircuitNode> ghost =
    extract_accessor<AccessorType::SOA<0>,CircuitNode>(regions[3]);

  int num_blocks = (p->num_wires+255) >> 8;

#ifdef TIME_CUDA_KERNELS
  cudaEvent_t ev_start, ev_end;
  CUDA_SAFE_CALL(cudaEventCreate(&ev_start, cudaEventDefault));
  CUDA_SAFE_CALL(cudaEventCreate(&ev_end, cudaEventDefault));
  CUDA_SAFE_CALL(cudaEventRecord(ev_start));
#endif
  calc_new_currents_kernel<<<num_blocks,256>>>(p->first_wire,
                                               p->num_wires,
					       p->dt,
					       p->steps,
                                               wires, pvt, owned, ghost);
#ifdef TIME_CUDA_KERNELS
  CUDA_SAFE_CALL(cudaEventRecord(ev_end));
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
#ifdef TIME_CUDA_KERNELS
  float ms;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, ev_start, ev_end));
  CUDA_SAFE_CALL(cudaEventDestroy(ev_start));
  CUDA_SAFE_CALL(cudaEventDestroy(ev_end));
  printf("CNC TIME = %f\n", ms);
#endif
#endif
}

template<typename AT1, typename AT2>
__global__
void distribute_charge_kernel(ptr_t first,
                              int num_wires,
			      float dt,
                              RegionAccessor<AT1, CircuitWire> wires,
                              RegionAccessor<AT1, CircuitNode> pvt,
                              RegionAccessor<AT2, CircuitNode> owned,
                              RegionAccessor<AT2, CircuitNode> ghost)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_wires)
  {
    ptr_t local_ptr = first + tid;

    CircuitWire &wire = wires.ref(local_ptr);

    reduce_local<GPUAccumulateCharge>(pvt, owned, ghost, wire.in_loc, wire.in_ptr, -dt * wire.current[0]);
    reduce_local<GPUAccumulateCharge>(pvt, owned, ghost, wire.out_loc, wire.out_ptr, dt * wire.current[WIRE_SEGMENTS-1]);
  }
}

__host__
void distribute_charge_gpu(CircuitPiece *p,
                           const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::SOA<0>, CircuitWire> wires = 
    extract_accessor<AccessorType::SOA<0>, CircuitWire>(regions[0]);
  RegionAccessor<AccessorType::SOA<0>, CircuitNode> pvt =
    extract_accessor<AccessorType::SOA<0>, CircuitNode>(regions[1]);
  RegionAccessor<AccessorType::ReductionFold<GPUAccumulateCharge>, CircuitNode> owned =
    extract_accessor<AccessorType::ReductionFold<GPUAccumulateCharge>, CircuitNode>(regions[2]);
  RegionAccessor<AccessorType::ReductionFold<GPUAccumulateCharge>, CircuitNode> ghost = 
    extract_accessor<AccessorType::ReductionFold<GPUAccumulateCharge>, CircuitNode>(regions[3]);
  int num_blocks = (p->num_wires+255) >> 8;

#ifdef TIME_CUDA_KERNELS
  cudaEvent_t ev_start, ev_end;
  CUDA_SAFE_CALL(cudaEventCreate(&ev_start, cudaEventDefault));
  CUDA_SAFE_CALL(cudaEventCreate(&ev_end, cudaEventDefault));
  CUDA_SAFE_CALL(cudaEventRecord(ev_start));
#endif
  distribute_charge_kernel<<<num_blocks,256>>>(p->first_wire,
                                               p->num_wires,
					       p->dt,
                                               wires, pvt, owned, ghost);
#ifdef TIME_CUDA_KERNELS
  CUDA_SAFE_CALL(cudaEventRecord(ev_end));
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
#ifdef TIME_CUDA_KERNELS
  float ms;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, ev_start, ev_end));
  CUDA_SAFE_CALL(cudaEventDestroy(ev_start));
  CUDA_SAFE_CALL(cudaEventDestroy(ev_end));
  printf("DC TIME = %f\n", ms);
#endif
#endif
}

template<typename AT>
__global__
void update_voltages_kernel(ptr_t first,
                            int num_nodes,
                            RegionAccessor<AT, CircuitNode> pvt,
                            RegionAccessor<AT, CircuitNode> owned,
                            RegionAccessor<AT, PointerLocation> locator)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_nodes)
  {
    ptr_t locator_ptr = first + tid;
    ptr_t local_node = first + tid;;
    // Figure out if this node is pvt or not
    {
      int is_pvt = locator.read(locator_ptr) == PRIVATE_PTR;
      if (is_pvt)
      {
        CircuitNode &cur_node = pvt.ref(local_node);
        // charge adds in, and then some leaks away
        cur_node.voltage += cur_node.charge / cur_node.capacitance;
        cur_node.voltage *= (1 - cur_node.leakage);
        cur_node.charge = 0;
      }
      else
      {
        CircuitNode &cur_node = owned.ref(local_node);
        // charge adds in, and then some leaks away
	if(cur_node.capacitance < 1e-10)
	  cur_node.capacitance = 1e-10;
        cur_node.voltage += cur_node.charge / cur_node.capacitance;
        cur_node.voltage *= (1 - cur_node.leakage);
        cur_node.charge = 0;
      }
    }
  }
}

__host__
void update_voltages_gpu(CircuitPiece *p,
                         const std::vector<PhysicalRegion> &regions)
{
#ifndef DISABLE_MATH
  RegionAccessor<AccessorType::SOA<0>, CircuitNode> pvt = 
    extract_accessor<AccessorType::SOA<0>, CircuitNode>(regions[0]);
  RegionAccessor<AccessorType::SOA<0>, CircuitNode> owned = 
    extract_accessor<AccessorType::SOA<0>, CircuitNode>(regions[1]);
  RegionAccessor<AccessorType::SOA<0>, PointerLocation> locator = 
    extract_accessor<AccessorType::SOA<0>, PointerLocation>(regions[2]);
  int num_blocks = (p->num_nodes+255) >> 8;

#ifdef TIME_CUDA_KERNELS
  cudaEvent_t ev_start, ev_end;
  CUDA_SAFE_CALL(cudaEventCreate(&ev_start, cudaEventDefault));
  CUDA_SAFE_CALL(cudaEventCreate(&ev_end, cudaEventDefault));
  CUDA_SAFE_CALL(cudaEventRecord(ev_start));
#endif
  update_voltages_kernel<<<num_blocks,256>>>(p->first_node,
                                             p->num_nodes,
                                             pvt, owned, locator);
#ifdef TIME_CUDA_KERNELS
  CUDA_SAFE_CALL(cudaEventRecord(ev_end));
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
#ifdef TIME_CUDA_KERNELS
  float ms;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, ev_start, ev_end));
  CUDA_SAFE_CALL(cudaEventDestroy(ev_start));
  CUDA_SAFE_CALL(cudaEventDestroy(ev_end));
  printf("UV TIME = %f\n", ms);
#endif
#endif
}


