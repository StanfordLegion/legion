/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#ifndef __KMEANS_H__
#define __KMEANS_H__

#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;

enum {
  TOP_LEVEL_TASK_ID,
  INITIALIZE_TASK_ID,
  KMEANS_ENERGY_TASK_ID,
  UPDATE_CENTERS_TASK_ID,
  CONVERGENCE_TASK_ID,
};

enum {
  CHUNK_TUNABLE,
  PREDICATION_DEPTH_TUNABLE,
};

enum {
  INT_SUM_REDUCTION = 1,
  DOUBLE_SUM_REDUCTION = 2,
};

class PointSet;
class CenterSet;

class InitializeTask : public TaskLauncher {
public:
  InitializeTask(const PointSet &points, const CenterSet &centers,
                 int *num_centers);
public:
  void dispatch(Context ctx, HighLevelRuntime *runtime);
public:
  static void cpu_variant(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime);
  static void register_variants(void);
};

class KmeansEnergyTask : public IndexLauncher {
public:
  KmeansEnergyTask(const PointSet &points, const CenterSet &centers,
                   const Predicate &pred);
public:
  Future dispatch(Context ctx, HighLevelRuntime *runtime);
public:
  static double cpu_variant(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
  static void register_variants(void);
};

class UpdateCentersTask : public TaskLauncher {
public:
  UpdateCentersTask(const CenterSet &centers, const Predicate &pred);
public:
  void dispatch(Context ctx, HighLevelRuntime *runtime);
public:
  static void cpu_variant(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime);
  static void register_variants(void);
};

class ConvergenceTask : public TaskLauncher {
public:
  ConvergenceTask(const Future &prev, const Future &next, const Predicate &pred);
public:
  Predicate dispatch(Context ctx, HighLevelRuntime *runtime);
public:
  static bool cpu_variant(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime);
  static void register_variants(void);
};

class PointSet {
public:
  enum Fields {
    FID_LOCATION,
  };
public:
  PointSet(Context ctx, HighLevelRuntime *runtime, int num_points);
  PointSet(const PointSet &rhs);
  ~PointSet(void);
public:
  PointSet& operator=(const PointSet &rhs);
public:
  inline LogicalRegion get_region(void) const { return handle; }
  inline LogicalPartition get_partition(void) const { return partition; }
  inline Domain get_domain(void) const { return color_domain; }
  void partition_set(int num_points, int num_chunks);
public:
  const Context ctx;
  HighLevelRuntime *const runtime;
protected:
  LogicalRegion handle;
  LogicalPartition partition;
  Domain color_domain;
};

class CenterSet {
public:
  enum Fields {
    FID_LOCATION,
    FID_PENDING_SUM,
    FID_PENDING_COUNT,
  };
public:
  CenterSet(Context ctx, HighLevelRuntime *runtime, int num_centers);
  CenterSet(const CenterSet &rhs);
  ~CenterSet(void);
public:
  CenterSet& operator=(const CenterSet &rhs);
public:
  inline LogicalRegion get_region(void) const { return handle; }
public:
  const Context ctx;
  HighLevelRuntime *const runtime;
protected:
  LogicalRegion handle;
};

class IntegerSum {
public:
  typedef int LHS;
  typedef int RHS;
  static const int identity;

  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);

  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

class DoubleSum {
public:
  typedef double LHS;
  typedef double RHS;
  static const double identity;

  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);

  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

class KmeansMapper : public DefaultMapper {
public:
  KmeansMapper(Machine machine, HighLevelRuntime *rt, Processor local);
public:
  virtual int get_tunable_value(const Task *task, TunableID tid, MappingTagID tag);
};

#endif // __KMEANS__
