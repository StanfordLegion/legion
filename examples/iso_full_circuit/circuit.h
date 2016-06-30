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


#ifndef __CIRCUIT_H__
#define __CIRCUIT_H__

#include <cmath>
#include <cstdio>
#include "legion.h"

//#define DISABLE_MATH

#define WIRE_SEGMENTS 10
#define STEPS         10000
#define DELTAT        1e-6

#define INDEX_TYPE    unsigned
#define INDEX_DIM     1

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

// Data type definitions

enum PointerLocation {
  PRIVATE_PTR,
  SHARED_PTR,
  GHOST_PTR,
};

enum {
  TOP_LEVEL_TASK_ID,
  CALC_NEW_CURRENTS_TASK_ID,
  DISTRIBUTE_CHARGE_TASK_ID,
  UPDATE_VOLTAGES_TASK_ID,
  CHECK_FIELD_TASK_ID,
  COPY_TASK_ID
};

enum {
  REDUCE_ID = 1,
};

enum NodeFields {
  FID_NODE_CAP,
  FID_LEAKAGE,
  FID_CHARGE,
  FID_NODE_VOLTAGE,
};

enum WireFields {
  FID_IN_PTR,
  FID_OUT_PTR,
  FID_IN_LOC,
  FID_OUT_LOC,
  FID_INDUCTANCE,
  FID_RESISTANCE,
  FID_WIRE_CAP,
  FID_CURRENT,
  FID_WIRE_VOLTAGE = (FID_CURRENT+WIRE_SEGMENTS),
  FID_LAST = (FID_WIRE_VOLTAGE+WIRE_SEGMENTS-1),
};

enum LocatorFields {
  FID_LOCATOR,
};

enum CircuitVariants {
  CIRCUIT_CPU_LEAF_VARIANT,
  CIRCUIT_GPU_LEAF_VARIANT,
};

struct Circuit {
  LogicalRegion all_nodes;
  LogicalRegion all_wires;
  LogicalRegion node_locator;
};

struct CircuitPiece {
  LogicalRegion pvt_nodes, shr_nodes, ghost_nodes;
  LogicalRegion pvt_wires;
  unsigned      num_wires;
  ptr_t         first_wire;
  unsigned      num_nodes;
  ptr_t         first_node;

  float         dt;
  int           steps;
};

struct Partitions {
  LogicalPartition pvt_wires;
  LogicalPartition pvt_nodes, shr_nodes, ghost_nodes;
  LogicalPartition node_locations;
};

// Reduction Op
class AccumulateCharge {
public:
  typedef float LHS;
  typedef float RHS;
  static const float identity;

  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);

  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

class CalcNewCurrentsTask : public IndexLauncher {
public:
  CalcNewCurrentsTask(LogicalPartition lp_pvt_wires,
                      LogicalPartition lp_pvt_nodes,
                      LogicalPartition lp_shr_nodes,
                      LogicalPartition lp_ghost_nodes,
                      LogicalRegion lr_all_wires,
                      LogicalRegion lr_all_nodes,
                      const Domain &launch_domain,
                      const ArgumentMap &arg_map);
public:
  bool launch_check_fields(Context ctx, HighLevelRuntime *runtime);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = CALC_NEW_CURRENTS_TASK_ID;
  static const bool CPU_BASE_LEAF = true;
  static const bool GPU_BASE_LEAF = true;
  static const int MAPPER_ID = 0;
protected:
  static bool dense_calc_new_currents(const CircuitPiece &piece,
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
                              RegionAccessor<AccessorType::Generic, float> *fa_voltage);
public:
  static void cpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime* rt);
#ifdef USE_CUDA
  static void gpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions);
#endif
};

class DistributeChargeTask : public IndexLauncher {
public:
  DistributeChargeTask(LogicalPartition lp_pvt_wires,
                       LogicalPartition lp_pvt_nodes,
                       LogicalPartition lp_shr_nodes,
                       LogicalPartition lp_ghost_nodes,
                       LogicalRegion lr_all_wires,
                       LogicalRegion lr_all_nodes,
                       const Domain &launch_domain,
                       const ArgumentMap &arg_map);
public:
  bool launch_check_fields(Context ctx, HighLevelRuntime *runtime);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = DISTRIBUTE_CHARGE_TASK_ID;
  static const bool CPU_BASE_LEAF = true;
  static const bool GPU_BASE_LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static void cpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime* rt);
#ifdef USE_CUDA
  static void gpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions);
#endif
};

class UpdateVoltagesTask : public IndexLauncher {
public:
  UpdateVoltagesTask(LogicalPartition lp_pvt_nodes,
                     LogicalPartition lp_shr_nodes,
                     LogicalPartition lp_node_locations,
                     LogicalRegion lr_all_nodes,
                     LogicalRegion lr_node_locator,
                     const Domain &launch_domain,
                     const ArgumentMap &arg_map);
public:
  bool launch_check_fields(Context ctx, HighLevelRuntime *runtime);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = UPDATE_VOLTAGES_TASK_ID;
  static const bool CPU_BASE_LEAF = true;
  static const bool GPU_BASE_LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static void cpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime* rt);
#ifdef USE_CUDA
  static void gpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions);
#endif
};

class CheckTask : public IndexLauncher {
public:
  CheckTask(LogicalPartition lp,
            LogicalRegion lr,
            FieldID fid,
            const Domain &launch_domain,
            const ArgumentMap &arg_map);
public:
  bool dispatch(Context ctx, HighLevelRuntime *runtime, bool success);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = CHECK_FIELD_TASK_ID;
  static const bool LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static bool cpu_impl(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime);
  static void register_task(void);
};

namespace TaskHelper {
  template<typename T>
  void dispatch_task(T &launcher, Context ctx, HighLevelRuntime *runtime,
                     bool perform_checks, bool &simulation_success, bool wait = false)
  {
    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    if (wait)
      fm.wait_all_results();
    // See if we need to perform any checks
    if (simulation_success && perform_checks)
    {
      simulation_success = launcher.launch_check_fields(ctx, runtime) && simulation_success;
      if (!simulation_success)
        printf("WARNING: First NaN values found in %s\n", T::TASK_NAME);
    }
  }

  template<typename T>
  void base_cpu_wrapper(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
  {
    const CircuitPiece *p = (CircuitPiece*)task->local_args;
    T::cpu_base_impl(*p, regions, ctx, runtime);
  }

#ifdef USE_CUDA
  template<typename T>
  void base_gpu_wrapper(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, HighLevelRuntime *runtime)
  {
    const CircuitPiece *p = (CircuitPiece*)task->local_args;
    T::gpu_base_impl(*p, regions); 
  }
#endif

  template<typename T>
  void register_cpu_variants(void)
  {
    HighLevelRuntime::register_legion_task<base_cpu_wrapper<T> >(T::TASK_ID, Processor::LOC_PROC,
                                                                 false/*single*/, true/*index*/,
                                                                 CIRCUIT_CPU_LEAF_VARIANT,
                                                                 TaskConfigOptions(T::CPU_BASE_LEAF),
                                                                 T::TASK_NAME);
  }

  template<typename T>
  void register_hybrid_variants(void)
  {
    HighLevelRuntime::register_legion_task<base_cpu_wrapper<T> >(T::TASK_ID, Processor::LOC_PROC,
                                                                 false/*single*/, true/*index*/,
                                                                 CIRCUIT_CPU_LEAF_VARIANT,
                                                                 TaskConfigOptions(T::CPU_BASE_LEAF),
                                                                 T::TASK_NAME);
#ifdef USE_CUDA
    HighLevelRuntime::register_legion_task<base_gpu_wrapper<T> >(T::TASK_ID, Processor::TOC_PROC,
                                                                 false/*single*/, true/*index*/,
                                                                 CIRCUIT_GPU_LEAF_VARIANT,
                                                                 TaskConfigOptions(T::GPU_BASE_LEAF),
                                                                 T::TASK_NAME);
#endif
  }
};

#endif // __CIRCUIT_H__

