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

using namespace Legion;

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
#ifndef SEQUENTIAL_LOAD_CIRCUIT
  INIT_NODES_TASK_ID,
  INIT_WIRES_TASK_ID,
  INIT_LOCATION_TASK_ID,
#endif
};

enum {
  REDUCE_ID = 1,
};

enum NodeFields {
  FID_NODE_CAP,
  FID_LEAKAGE,
  FID_CHARGE,
  FID_NODE_VOLTAGE,
  FID_PIECE_COLOR,
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

typedef FieldAccessor<READ_ONLY,float,1,coord_t,Realm::AffineAccessor<float,1,coord_t> > AccessorROfloat;
typedef FieldAccessor<READ_WRITE,float,1,coord_t,Realm::AffineAccessor<float,1,coord_t> > AccessorRWfloat;
typedef FieldAccessor<WRITE_ONLY,float,1,coord_t,Realm::AffineAccessor<float,1,coord_t> > AccessorWOfloat;

typedef FieldAccessor<READ_ONLY,Point<1>,1,coord_t,Realm::AffineAccessor<Point<1>,1,coord_t> > AccessorROpoint;
typedef FieldAccessor<READ_WRITE,Point<1>,1,coord_t,Realm::AffineAccessor<Point<1>,1,coord_t> > AccessorRWpoint;
typedef FieldAccessor<WRITE_ONLY,Point<1>,1,coord_t,Realm::AffineAccessor<Point<1>,1,coord_t> > AccessorWOpoint;

typedef FieldAccessor<READ_ONLY,PointerLocation,1,coord_t,Realm::AffineAccessor<PointerLocation,1,coord_t> > AccessorROloc;
typedef FieldAccessor<READ_WRITE,PointerLocation,1,coord_t,Realm::AffineAccessor<PointerLocation,1,coord_t> > AccessorRWloc;
typedef FieldAccessor<WRITE_ONLY,PointerLocation,1,coord_t,Realm::AffineAccessor<PointerLocation,1,coord_t> > AccessorWOloc;

struct Circuit {
  LogicalRegion all_nodes;
  LogicalRegion all_wires;
  LogicalRegion node_locator;
};

struct CircuitPiece {
  LogicalRegion pvt_nodes, shr_nodes, ghost_nodes;
  LogicalRegion pvt_wires;
  unsigned      num_wires;
  Point<1>      first_wire;
  unsigned      num_nodes;
  Point<1>      first_node;

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
  bool launch_check_fields(Context ctx, Runtime *runtime);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = CALC_NEW_CURRENTS_TASK_ID;
  static const bool CPU_BASE_LEAF = true;
  static const bool GPU_BASE_LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static void cpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* rt);
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
  bool launch_check_fields(Context ctx, Runtime *runtime);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = DISTRIBUTE_CHARGE_TASK_ID;
  static const bool CPU_BASE_LEAF = true;
  static const bool GPU_BASE_LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static void cpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* rt);
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
  bool launch_check_fields(Context ctx, Runtime *runtime);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = UPDATE_VOLTAGES_TASK_ID;
  static const bool CPU_BASE_LEAF = true;
  static const bool GPU_BASE_LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static void cpu_base_impl(const CircuitPiece &piece,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime* rt);
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
  bool dispatch(Context ctx, Runtime *runtime, bool success);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = CHECK_FIELD_TASK_ID;
  static const bool LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static bool cpu_impl(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime);
  static void register_task(void);
};

#ifndef SEQUENTIAL_LOAD_CIRCUIT
class InitNodesTask : public IndexLauncher {
public:
  InitNodesTask(LogicalRegion lr_all_nodes,
                LogicalPartition lp_equal_nodes,
                IndexSpace launch_space);
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = INIT_NODES_TASK_ID;
  static const bool LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static void cpu_base_impl(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void register_task(void);
};

class InitWiresTask : public IndexLauncher {
public:
  struct Args {
  public:
    Args(int p, int n, int pct)
      : num_pieces(p), nodes_per_piece(n), pct_wire_in_piece(pct) { }
  public:
    int num_pieces;
    int nodes_per_piece;
    int pct_wire_in_piece;
  };
public:
  InitWiresTask(LogicalRegion lr_all_wires,
                LogicalPartition lp_equal_wires,
                IndexSpace launch_space,
                int num_pieces, int nodes_per_piece,
                int pct_wire_in_piece);
protected:
  Args args;
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = INIT_WIRES_TASK_ID;
  static const bool LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static void cpu_base_impl(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void register_task(void);
};

class InitLocationTask : public IndexLauncher {
public:
  struct Args {
  public:
    Args(LogicalPartition p, LogicalPartition s)
      : lp_private(p), lp_shared(s) { }
  public:
    LogicalPartition lp_private;
    LogicalPartition lp_shared;
  };
public:
  InitLocationTask(LogicalRegion lr_location,
                   LogicalPartition lp_equal_location,
                   LogicalRegion lr_all_wires,
                   LogicalPartition lp_equal_wires,
                   IndexSpace launch_space,
                   LogicalPartition lp_private,
                   LogicalPartition lp_shared);
protected:
  Args args;
public:
  static const char * const TASK_NAME;
  static const int TASK_ID = INIT_LOCATION_TASK_ID;
  static const bool LEAF = true;
  static const int MAPPER_ID = 0;
public:
  static void cpu_base_impl(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void register_task(void);
};
#endif

namespace TaskHelper {
  template<typename T>
  void dispatch_task(T &launcher, Context ctx, Runtime *runtime,
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
                        Context ctx, Runtime *runtime)
  {
    const CircuitPiece *p = (CircuitPiece*)task->local_args;
    T::cpu_base_impl(*p, regions, ctx, runtime);
  }

#ifdef USE_CUDA
  template<typename T>
  void base_gpu_wrapper(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
  {
    const CircuitPiece *p = (CircuitPiece*)task->local_args;
    T::gpu_base_impl(*p, regions); 
  }
#endif

  template<typename T>
  void register_hybrid_variants(void)
  {
    {
      TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf(T::CPU_BASE_LEAF);
      Runtime::preregister_task_variant<base_cpu_wrapper<T> >(registrar, T::TASK_NAME);
    }

#ifdef USE_CUDA
    {
      TaskVariantRegistrar registrar(T::TASK_ID, T::TASK_NAME);
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf(T::GPU_BASE_LEAF);
      Runtime::preregister_task_variant<base_gpu_wrapper<T> >(registrar, T::TASK_NAME);
    }
#endif
  }
};

#endif // __CIRCUIT_H__

