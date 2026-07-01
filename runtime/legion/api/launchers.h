/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_LAUNCHERS_H__
#define __LEGION_LAUNCHERS_H__

#include "legion/api/argument_map.h"
#include "legion/api/constraints.h"
#include "legion/api/future_map.h"
#include "legion/api/interop.h"
#include "legion/api/predicate.h"
#include "legion/api/requirements.h"
#include "legion/api/sync.h"

namespace Legion {

  /**
   * \struct StaticDependence
   * This is a helper class for specifying static dependences
   * between operations launched by an application. Operations
   * can optionally specify these dependences to aid in reducing
   * runtime overhead. These static dependences need only
   * be specified for dependences based on region requirements.
   */
  struct StaticDependence : public Unserializable {
  public:
    StaticDependence(void);
    StaticDependence(
        unsigned previous_offset, unsigned previous_req_index,
        unsigned current_req_index, DependenceType dtype,
        bool validates = false, bool shard_only = false);
  public:
    inline void add_field(FieldID fid);
  public:
    // The relative offset from this operation to
    // previous operation in the stream of operations
    // (e.g. 1 is the operation launched immediately before)
    unsigned previous_offset;
    // Region requirement of the previous operation for the dependence
    unsigned previous_req_index;
    // Region requirement of the current operation for the dependence
    unsigned current_req_index;
    // The type of the dependence
    DependenceType dependence_type;
    // Whether this requirement validates the previous writer
    bool validates;
    // Whether this dependence is a shard-only dependence for
    // control replication or it depends on all other copies
    bool shard_only;
    // Fields that have the dependence
    std::set<FieldID> dependent_fields;
  };

  /**
   * \struct TaskLauncher
   * Task launchers are objects that describe a launch
   * configuration to the runtime.  They can be re-used
   * and safely modified between calls to task launches.
   * @see Runtime
   */
  struct TaskLauncher {
  public:
    TaskLauncher(void);
    TaskLauncher(
        TaskID tid, UntypedBuffer arg = UntypedBuffer(),
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
  public:
    inline IndexSpaceRequirement& add_index_requirement(
        const IndexSpaceRequirement& req);
    inline RegionRequirement& add_region_requirement(
        const RegionRequirement& req);
    inline void add_field(unsigned idx, FieldID fid, bool inst = true);
  public:
    inline void add_future(Future f);
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier bar);
    inline void add_arrival_barrier(PhaseBarrier bar);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    inline void set_predicate_false_future(Future f);
    inline void set_predicate_false_result(UntypedBuffer arg);
  public:
    inline void set_independent_requirements(bool independent);
  public:
    TaskID task_id;
    std::vector<IndexSpaceRequirement> index_requirements;
    std::vector<RegionRequirement> region_requirements;
    std::vector<Future> futures;
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    UntypedBuffer argument;
    Predicate predicate;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
    DomainPoint point;
    // Only used in control replication contexts for
    // doing sharding. If left unspecified the runtime
    // will use an index space of size 1 containing 'point'
    IndexSpace sharding_space;
  public:
    // If the predicate is set to anything other than
    // Predicate::TRUE_PRED, then the application must
    // specify a value for the future in the case that
    // the predicate resolves to false. UntypedBuffer(nullptr,0)
    // can be used if the task's return type is void.
    Future predicate_false_future;
    UntypedBuffer predicate_false_result;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    // Users can tell the runtime this task is eligible
    // for inlining by the mapper. This will invoke the
    // select_task_options call inline as part of the launch
    // logic for this task to allow the mapper to decide
    // whether to inline the task or not. Note that if the
    // mapper pre-empts during execution then resuming it
    // may take a long time if another long running task
    // gets scheduled on the processor that launched this task.
    bool enable_inlining;
  public:
    // In some cases users (most likely compilers) will want
    // to run a light-weight function (e.g. a continuation)
    // as a task that just depends on futures once those futures
    // are ready on a local processor where the parent task
    // is executing. We call this a local function task and it
    // must not have any region requirements. It must als be a
    // pure function with no side effects. This task will
    // not have the option of being distributed to remote nodes.
    bool local_function_task;
  public:
    // Users can inform the runtime that all region requirements
    // are independent of each other in this task. Independent
    // means that either field sets are independent or region
    // requirements are disjoint based on the region tree.
    bool independent_requirements;
  public:
    // Instruct the runtime that it does not need to produce
    // a future or future map result for this index task
    bool elide_future_return;
    // Provide an optional future return size. In general you
    // shouldn't need to use this and should prefer specifying
    // the future return size when you register a task variant.
    std::optional<size_t> future_return_size;
  public:
    bool silence_warnings;
  };

  /**
   * \struct IndexTaskLauncher
   * Index launchers are objects that describe the launch
   * of an index space of tasks to the runtime.  They can
   * be re-used and safely modified between calls to
   * index space launches.
   * @see Runtime
   */
  struct IndexTaskLauncher {
  public:
    IndexTaskLauncher(void);
    IndexTaskLauncher(
        TaskID tid, Domain domain, UntypedBuffer global_arg = UntypedBuffer(),
        ArgumentMap map = ArgumentMap(), Predicate pred = Predicate::TRUE_PRED,
        bool must = false, MapperID id = 0, MappingTagID tag = 0,
        UntypedBuffer map_arg = UntypedBuffer(), const char* provenance = "");
    IndexTaskLauncher(
        TaskID tid, IndexSpace launch_space,
        UntypedBuffer global_arg = UntypedBuffer(),
        ArgumentMap map = ArgumentMap(), Predicate pred = Predicate::TRUE_PRED,
        bool must = false, MapperID id = 0, MappingTagID tag = 0,
        UntypedBuffer map_arg = UntypedBuffer(), const char* provenance = "");
  public:
    inline IndexSpaceRequirement& add_index_requirement(
        const IndexSpaceRequirement& req);
    inline RegionRequirement& add_region_requirement(
        const RegionRequirement& req);
    inline void add_field(unsigned idx, FieldID fid, bool inst = true);
  public:
    inline void add_future(Future f);
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier bar);
    inline void add_arrival_barrier(PhaseBarrier bar);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    inline void set_predicate_false_future(Future f);
    inline void set_predicate_false_result(UntypedBuffer arg);
  public:
    inline void set_independent_requirements(bool independent);
  public:
    TaskID task_id;
    Domain launch_domain;
    IndexSpace launch_space;
    // Will only be used in control replication context. If left
    // unset the runtime will use launch_domain/launch_space
    IndexSpace sharding_space;
    std::vector<IndexSpaceRequirement> index_requirements;
    std::vector<RegionRequirement> region_requirements;
    std::vector<Future> futures;
    // These are appended to the futures for the point
    // task after the futures sent to all points above
    std::vector<ArgumentMap> point_futures;
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    UntypedBuffer global_arg;
    ArgumentMap argument_map;
    Predicate predicate;
    // Specify that all the point tasks in this index launch be
    // able to run concurrently, meaning they all must map to
    // different processors and they cannot have interfering region
    // requirements that might lead to dependences. Note that the
    // runtime guarantees that concurrent index launches will not
    // deadlock with other concurrent index launches which requires
    // additional analysis. Currently concurrent index space launches
    // will only be allowed to map to leaf task variants currently.
    ConcurrentID concurrent_functor;  // = 0
    bool concurrent;                  // = false
    // This will convert this index space launch into a must
    // epoch launch which supports interfering region requirements
    bool must_parallelism;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
  public:
    // If the predicate is set to anything other than
    // Predicate::TRUE_PRED, then the application must
    // specify a value for the future in the case that
    // the predicate resolves to false. UntypedBuffer(nullptr,0)
    // can be used if the task's return type is void.
    Future predicate_false_future;
    UntypedBuffer predicate_false_result;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    // Users can tell the runtime this task is eligible
    // for inlining by the mapper. This will invoke the
    // select_task_options call inline as part of the launch
    // logic for this task to allow the mapper to decide
    // whether to inline the task or not. Note that if the
    // mapper pre-empts during execution then resuming it
    // may take a long time if another long running task
    // gets scheduled on the processor that launched this task.
    bool enable_inlining;
  public:
    // Users can inform the runtime that all region requirements
    // are independent of each other in this task. Independent
    // means that either field sets are independent or region
    // requirements are disjoint based on the region tree.
    bool independent_requirements;
  public:
    bool silence_warnings;
  public:
    // Instruct the runtime that it does not need to produce
    // a future or future map result for this index task
    bool elide_future_return;
    // Provide an optional future return size. In general you
    // shouldn't need to use this and should prefer specifying
    // the future return size when you register a task variant.
    std::optional<size_t> future_return_size;
  public:
    // Initial value for reduction
    Future initial_value;
  };

  /**
   * \struct InlineLauncher
   * Inline launchers are objects that describe the launch
   * of an inline mapping operation to the runtime.  They
   * can be re-used and safely modified between calls to
   * inline mapping operations.
   * @see Runtime
   */
  struct InlineLauncher {
  public:
    InlineLauncher(void);
    InlineLauncher(
        const RegionRequirement& req, MapperID id = 0, MappingTagID tag = 0,
        LayoutConstraintID layout_id = 0,
        UntypedBuffer map_arg = UntypedBuffer(), const char* provenance = "");
  public:
    inline void add_field(FieldID fid, bool inst = true);
  public:
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier bar);
    inline void add_arrival_barrier(PhaseBarrier bar);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    RegionRequirement requirement;
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
  public:
    LayoutConstraintID layout_constraint_id;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  };

  /**
   * \struct CopyLauncher
   * Copy launchers are objects that can be used to issue
   * copies between two regions including regions that are
   * not of the same region tree.  Copy operations specify
   * an arbitrary number of pairs of source and destination
   * region requirements.  The source region requirements
   * must be READ_ONLY, while the destination requirements
   * must be either READ_WRITE, WRITE_ONLY, or REDUCE with
   * a reduction function.  While the regions in a source
   * and a destination pair do not have to be in the same
   * region tree, one of the following two conditions must hold:
   * 1. The two regions share an index space tree and the
   *    source region's index space is an ancestor of the
   *    destination region's index space.
   * 2. The source and destination index spaces must be
   *    of the same kind (either dimensions match or number
   *    of elements match in the element mask) and the source
   *    region's index space must dominate the destination
   *    region's index space.
   * If either of these two conditions does not hold then
   * the runtime will issue an error.
   * @see Runtime
   */
  struct CopyLauncher {
  public:
    CopyLauncher(
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
  public:
    inline unsigned add_copy_requirements(
        const RegionRequirement& src, const RegionRequirement& dst);
    inline void add_src_field(unsigned idx, FieldID fid, bool inst = true);
    inline void add_dst_field(unsigned idx, FieldID fid, bool inst = true);
  public:
    // Specify src/dst indirect requirements (must have exactly 1 field)
    inline void add_src_indirect_field(
        FieldID src_idx_fid, const RegionRequirement& src_idx_req,
        bool is_range_indirection = false, bool inst = true);
    inline void add_dst_indirect_field(
        FieldID dst_idx_fid, const RegionRequirement& dst_idx_req,
        bool is_range_indirection = false, bool inst = true);
    inline RegionRequirement& add_src_indirect_field(
        const RegionRequirement& src_idx_req,
        bool is_range_indirection = false);
    inline RegionRequirement& add_dst_indirect_field(
        const RegionRequirement& dst_idx_req,
        bool is_range_indirection = false);
  public:
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier bar);
    inline void add_arrival_barrier(PhaseBarrier bar);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    std::vector<RegionRequirement> src_requirements;
    std::vector<RegionRequirement> dst_requirements;
    std::vector<RegionRequirement> src_indirect_requirements;
    std::vector<RegionRequirement> dst_indirect_requirements;
    std::vector<bool> src_indirect_is_range;
    std::vector<bool> dst_indirect_is_range;
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    Predicate predicate;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
    DomainPoint point;
    // Only used in control replication contexts for
    // doing sharding. If left unspecified the runtime
    // will use an index space of size 1 containing 'point'
    IndexSpace sharding_space;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    // Whether the source and destination indirections can lead
    // to out-of-range access into the instances to skip
    bool possible_src_indirect_out_of_range;
    bool possible_dst_indirect_out_of_range;
    // Whether the destination indirection can lead to aliasing
    // in the destination instances requiring synchronization
    bool possible_dst_indirect_aliasing;
  public:
    bool silence_warnings;
  };

  /**
   * \struct IndexCopyLauncher
   * An index copy launcher is the same as a normal copy launcher
   * but it supports the ability to launch multiple copies all
   * over a single index space domain. This means that region
   * requirements can use projection functions the same as with
   * index task launches.
   * @see CopyLauncher
   * @see Runtime
   */
  struct IndexCopyLauncher {
  public:
    IndexCopyLauncher(void);
    IndexCopyLauncher(
        Domain domain, Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
    IndexCopyLauncher(
        IndexSpace space, Predicate pred = Predicate::TRUE_PRED,
        MapperID id = 0, MappingTagID tag = 0,
        UntypedBuffer map_arg = UntypedBuffer(), const char* provenance = "");
  public:
    inline unsigned add_copy_requirements(
        const RegionRequirement& src, const RegionRequirement& dst);
    inline void add_src_field(unsigned idx, FieldID fid, bool inst = true);
    inline void add_dst_field(unsigned idx, FieldID fid, bool inst = true);
  public:
    // Specify src/dst indirect requirements (must have exactly 1 field)
    inline void add_src_indirect_field(
        FieldID src_idx_fid, const RegionRequirement& src_idx_req,
        bool is_range_indirection = false, bool inst = true);
    inline void add_dst_indirect_field(
        FieldID dst_idx_fid, const RegionRequirement& dst_idx_req,
        bool is_range_indirection = false, bool inst = true);
    inline RegionRequirement& add_src_indirect_field(
        const RegionRequirement& src_idx_req,
        bool is_range_indirection = false);
    inline RegionRequirement& add_dst_indirect_field(
        const RegionRequirement& dst_idx_req,
        bool is_range_indirection = false);
  public:
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier bar);
    inline void add_arrival_barrier(PhaseBarrier bar);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    std::vector<RegionRequirement> src_requirements;
    std::vector<RegionRequirement> dst_requirements;
    std::vector<RegionRequirement> src_indirect_requirements;
    std::vector<RegionRequirement> dst_indirect_requirements;
    std::vector<bool> src_indirect_is_range;
    std::vector<bool> dst_indirect_is_range;
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    Domain launch_domain;
    IndexSpace launch_space;
    // Will only be used in control replication context. If left
    // unset the runtime will use launch_domain/launch_space
    IndexSpace sharding_space;
    Predicate predicate;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    // Whether the source and destination indirections can lead
    // to out-of-range access into the instances to skip
    bool possible_src_indirect_out_of_range;
    bool possible_dst_indirect_out_of_range;
    // Whether the destination indirection can lead to aliasing
    // in the destination instances requiring synchronization
    bool possible_dst_indirect_aliasing;
    // Whether the individual point copies should operate collectively
    // together in the case of indirect copies (e.g. allow indirections
    // to refer to instances from other points). These settings have
    // no effect in the case of copies without indirections.
    bool collective_src_indirect_points;
    bool collective_dst_indirect_points;
  public:
    bool silence_warnings;
  };

  /**
   * \struct FillLauncher
   * Fill launchers are objects that describe the parameters
   * for issuing a fill operation.
   * @see Runtime
   */
  struct FillLauncher {
  public:
    FillLauncher(void);
    FillLauncher(
        LogicalRegion handle, LogicalRegion parent, UntypedBuffer arg,
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
    FillLauncher(
        LogicalRegion handle, LogicalRegion parent, Future f,
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
  public:
    inline void set_argument(UntypedBuffer arg);
    inline void set_future(Future f);
    inline void add_field(FieldID fid);
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier bar);
    inline void add_arrival_barrier(PhaseBarrier bar);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    LogicalRegion handle;
    LogicalRegion parent;
    UntypedBuffer argument;
    Future future;
    Predicate predicate;
    std::set<FieldID> fields;
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
    DomainPoint point;
    // Only used in control replication contexts for
    // doing sharding. If left unspecified the runtime
    // will use an index space of size 1 containing 'point'
    IndexSpace sharding_space;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    bool silence_warnings;
  };

  /**
   * \struct IndexFillLauncher
   * Index fill launchers are objects that are used to describe
   * a fill over a particular domain. They can be used with
   * projeciton functions to describe a fill over an arbitrary
   * set of logical regions.
   * @see FillLauncher
   * @see Runtime
   */
  struct IndexFillLauncher {
  public:
    IndexFillLauncher(void);
    // Region projection
    IndexFillLauncher(
        Domain domain, LogicalRegion handle, LogicalRegion parent,
        UntypedBuffer arg, ProjectionID projection = 0,
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
    IndexFillLauncher(
        Domain domain, LogicalRegion handle, LogicalRegion parent, Future f,
        ProjectionID projection = 0, Predicate pred = Predicate::TRUE_PRED,
        MapperID id = 0, MappingTagID tag = 0,
        UntypedBuffer map_arg = UntypedBuffer(), const char* provenance = "");
    IndexFillLauncher(
        IndexSpace space, LogicalRegion handle, LogicalRegion parent,
        UntypedBuffer arg, ProjectionID projection = 0,
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
    IndexFillLauncher(
        IndexSpace space, LogicalRegion handle, LogicalRegion parent, Future f,
        ProjectionID projection = 0, Predicate pred = Predicate::TRUE_PRED,
        MapperID id = 0, MappingTagID tag = 0,
        UntypedBuffer map_arg = UntypedBuffer(), const char* provenance = "");
    // Partition projection
    IndexFillLauncher(
        Domain domain, LogicalPartition handle, LogicalRegion parent,
        UntypedBuffer arg, ProjectionID projection = 0,
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
    IndexFillLauncher(
        Domain domain, LogicalPartition handle, LogicalRegion parent, Future f,
        ProjectionID projection = 0, Predicate pred = Predicate::TRUE_PRED,
        MapperID id = 0, MappingTagID tag = 0,
        UntypedBuffer map_arg = UntypedBuffer(), const char* provenance = "");
    IndexFillLauncher(
        IndexSpace space, LogicalPartition handle, LogicalRegion parent,
        UntypedBuffer arg, ProjectionID projection = 0,
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
    IndexFillLauncher(
        IndexSpace space, LogicalPartition handle, LogicalRegion parent,
        Future f, ProjectionID projection = 0,
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
  public:
    inline void set_argument(UntypedBuffer arg);
    inline void set_future(Future f);
    inline void add_field(FieldID fid);
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier bar);
    inline void add_arrival_barrier(PhaseBarrier bar);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    Domain launch_domain;
    IndexSpace launch_space;
    // Will only be used in control replication context. If left
    // unset the runtime will use launch_domain/launch_space
    IndexSpace sharding_space;
    LogicalRegion region;
    LogicalPartition partition;
    LogicalRegion parent;
    ProjectionID projection;
    UntypedBuffer argument;
    Future future;
    Predicate predicate;
    std::set<FieldID> fields;
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    bool silence_warnings;
  };

  /**
   * \struct DiscardLauncher
   * Discard launchers will reset the state of one or more fields
   * for a particular logical region to an uninitialized state.
   * @see Runtime
   */
  struct DiscardLauncher {
  public:
    DiscardLauncher(LogicalRegion handle, LogicalRegion parent);
  public:
    inline void add_field(FieldID fid);
  public:
    LogicalRegion handle;
    LogicalRegion parent;
    std::set<FieldID> fields;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    bool silence_warnings;
  };

  /**
   * \struct AttachLauncher
   * Attach launchers are used for attaching existing physical resources
   * outside of a Legion application to a specific logical region.
   * This can include attaching files or arrays from inter-operating
   * programs. We provide a generic attach launcher than can handle
   * all kinds of attachments. Each attach launcher should be used
   * for attaching only one kind of resource. Resources are described
   * using Realm::ExternalInstanceResource descriptors (interface can
   * be found in realm/instance.h). There are many different kinds
   * of external instance resource descriptors including:
   * - Realm::ExternalMemoryResource for host pointers (realm/instance.h)
   * - Realm::ExternalFileResource for POSIX files (realm/instance.h)
   * - Realm::ExternalCudaMemoryResource for CUDA pointers
   * (realm/cuda/cuda_access.h)
   * - Realm::ExternalHipMemoryResource for HIP pointers
   * (realm/hip/hip_access.h)
   * - Realm::ExternalHDF5Resource for HDF5 files (realm/hdf5/hdf5_access.h)
   * ...
   * Please explore the Realm code base for all the different kinds of
   * external resources that you can attach to logical regions.
   * @see Runtime
   */
  struct AttachLauncher {
  public:
    AttachLauncher(
        ExternalResource resource, LogicalRegion handle, LogicalRegion parent,
        const bool restricted = true, const bool mapped = true);
    // Declared here to avoid superfluous compiler warnings
    // Can be remove after deprecated members are removed
    ~AttachLauncher(void);
  public:
    inline void initialize_constraints(
        bool column_major, bool soa, const std::vector<FieldID>& fields,
        const std::map<FieldID, size_t>* alignments = nullptr);
    LEGION_DEPRECATED("Use Realm::ExternalFileResource instead")
    inline void attach_file(
        const char* file_name, const std::vector<FieldID>& fields,
        LegionFileMode mode);
    LEGION_DEPRECATED("Use Realm::ExternalHDF5Resource instead")
    inline void attach_hdf5(
        const char* file_name, const std::map<FieldID, const char*>& field_map,
        LegionFileMode mode);
    // Helper methods for AOS and SOA arrays, but it is totally
    // acceptable to fill in the layout constraint set manually
    LEGION_DEPRECATED("Use Realm::ExternalMemoryResource instead")
    inline void attach_array_aos(
        void* base, bool column_major, const std::vector<FieldID>& fields,
        Memory memory = Memory::NO_MEMORY,
        const std::map<FieldID, size_t>* alignments = nullptr);
    LEGION_DEPRECATED("Use Realm::ExternalMemoryResource instead")
    inline void attach_array_soa(
        void* base, bool column_major, const std::vector<FieldID>& fields,
        Memory memory = Memory::NO_MEMORY,
        const std::map<FieldID, size_t>* alignments = nullptr);
  public:
    ExternalResource resource;
    LogicalRegion parent;
    LogicalRegion handle;
    std::set<FieldID> privilege_fields;
  public:
    // This will be cloned each time you perform an attach with this launcher
    const Realm::ExternalInstanceResource* external_resource;
  public:
    LayoutConstraintSet constraints;
  public:
    // Whether this instance will be restricted when attached
    bool restricted /*= true*/;
    // Whether this region should be mapped by the calling task
    bool mapped; /*= true*/
    // Only matters for control replicated parent tasks
    // Indicate whether all the shards are providing the same data
    // or whether they are each providing different data
    // Collective means that each shard provides its own copy of the
    // data and non-collective means every shard provides the same data
    // Defaults to 'true' for external instances and 'false' for files
    bool collective;
    // For collective cases, indicate whether the runtime should
    // deduplicate data across shards in the same process
    // This is useful for cases where there is one file or external
    // instance per process but multiple shards per process
    bool deduplicate_across_shards;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Data for files
    LEGION_DEPRECATED("file_name is deprecated, use external_resource")
    const char* file_name;
    LEGION_DEPRECATED("mode is deprecated, use external_resource")
    LegionFileMode mode;
    LEGION_DEPRECATED("file_fields is deprecated, use external_resource")
    std::vector<FieldID> file_fields;  // normal files
    // This member must still be populated if you're attaching to an HDF5 file
    std::map<FieldID, /*file name*/ const char*> field_files;  // hdf5 files
  public:
    // Optional footprint of the instance in memory in bytes
    size_t footprint;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  };

  /**
   * \struct IndexAttachLauncher
   * An index attach launcher allows the application to attach
   * many external resources concurrently to different subregions
   * of a common region tree. For more information regarding what
   * kinds of external resources can be attached please see the
   * documentation for AttachLauncher.
   * @see AttachLauncher
   * @see Runtime
   */
  struct IndexAttachLauncher {
  public:
    IndexAttachLauncher(
        ExternalResource resource, LogicalRegion parent,
        const bool restricted = true);
    // Declared here to avoid superfluous compiler warnings
    // Can be remove after deprecated members are removed
    ~IndexAttachLauncher(void);
  public:
    inline void initialize_constraints(
        bool column_major, bool soa, const std::vector<FieldID>& fields,
        const std::map<FieldID, size_t>* alignments = nullptr);
    inline void add_external_resource(
        LogicalRegion handle, const Realm::ExternalInstanceResource* resource);
    LEGION_DEPRECATED("Use Realm::ExternalFileResource instead")
    inline void attach_file(
        LogicalRegion handle, const char* file_name,
        const std::vector<FieldID>& fields, LegionFileMode mode);
    LEGION_DEPRECATED("Use Realm::ExternalHDF5Resource instead")
    inline void attach_hdf5(
        LogicalRegion handle, const char* file_name,
        const std::map<FieldID, const char*>& field_map, LegionFileMode mode);
    // Helper methods for AOS and SOA arrays, but it is totally
    // acceptable to fill in the layout constraint set manually
    LEGION_DEPRECATED("Use Realm::ExternalMemoryResource instead")
    inline void attach_array_aos(
        LogicalRegion handle, void* base, bool column_major,
        const std::vector<FieldID>& fields, Memory memory = Memory::NO_MEMORY,
        const std::map<FieldID, size_t>* alignments = nullptr);
    LEGION_DEPRECATED("Use Realm::ExternalMemoryResource instead")
    inline void attach_array_soa(
        LogicalRegion handle, void* base, bool column_major,
        const std::vector<FieldID>& fields, Memory memory = Memory::NO_MEMORY,
        const std::map<FieldID, size_t>* alignments = nullptr);
  public:
    ExternalResource resource;
    LogicalRegion parent;
    std::set<FieldID> privilege_fields;
    std::vector<LogicalRegion> handles;
    // This is the vector external resource objects that are going to
    // attached to the vector of logical region handles
    // These will be cloned each time you perform an attach with this launcher
    std::vector<const Realm::ExternalInstanceResource*> external_resources;
  public:
    LayoutConstraintSet constraints;
  public:
    // Whether these instances will be restricted when attached
    bool restricted /*= true*/;
    // Whether the runtime should check for duplicate resources across
    // the shards in a control replicated context, it is illegal to pass
    // in the same resource to different shards if this is set to false
    bool deduplicate_across_shards;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Data for files
    LEGION_DEPRECATED("mode is deprecated, use external_resources")
    LegionFileMode mode;
    LEGION_DEPRECATED("file_names is deprecated, use external_resources")
    std::vector<const char*> file_names;
    LEGION_DEPRECATED("file_fields is deprecated, use external_resources")
    std::vector<FieldID> file_fields;  // normal files
    // This data structure must still be filled in for using HDF5 files
    std::map<FieldID, std::vector</*file name*/ const char*> >
        field_files;  // hdf5 files
  public:
    // Data for external instances
    LEGION_DEPRECATED("pointers is deprecated, use external_resources")
    std::vector<PointerConstraint> pointers;
  public:
    // Optional footprint of the instance in memory in bytes
    // You only need to fill this in when using depcreated fields
    std::vector<size_t> footprint;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  };

  /**
   * \struct PredicateLauncher
   * Predicate launchers are used for merging several predicates
   * into a new predicate either by an 'AND' or an 'OR' operation.
   * @see Runtime
   */
  struct PredicateLauncher {
  public:
    explicit PredicateLauncher(bool and_op = true);
  public:
    inline void add_predicate(const Predicate& pred);
  public:
    bool and_op;  // if not 'and' then 'or'
    std::vector<Predicate> predicates;
    std::string provenance;
  };

  /**
   * \struct TimingLauncher
   * Timing launchers are used for issuing a timing measurement.
   * @see Runtime
   */
  struct TimingLauncher {
  public:
    TimingLauncher(TimingMeasurement measurement);
  public:
    inline void add_precondition(const Future& f);
  public:
    TimingMeasurement measurement;
    std::set<Future> preconditions;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  };

  /**
   * \struct TunableLauncher
   * Tunable launchers are used for requesting tunable values from mappers.
   * @see Runtime
   */
  struct TunableLauncher {
  public:
    TunableLauncher(
        TunableID tid, MapperID mapper = 0, MappingTagID tag = 0,
        size_t return_type_size = SIZE_MAX);
  public:
    TunableID tunable;
    MapperID mapper;
    MappingTagID tag;
    UntypedBuffer arg;
    std::vector<Future> futures;
    size_t return_type_size;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  };

  /**
   * \struct AcquireLauncher
   * An AcquireLauncher is a class that is used for supporting user-level
   * software coherence when working with logical regions held in
   * simultaneous coherence mode.  By default simultaneous mode requires
   * all users to use the same physical instance.  By acquiring coherence
   * on the physical region, a parent task can launch sub-tasks which
   * are not required to use the same physical instance.  Synchronization
   * primitives are allowed to specify what must occur before the
   * acquire operation is performed.
   */
  struct AcquireLauncher {
  public:
    AcquireLauncher(
        LogicalRegion logical_region, LogicalRegion parent_region,
        PhysicalRegion physical_region = PhysicalRegion(),
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
  public:
    inline void add_field(FieldID f);
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier pb);
    inline void add_arrival_barrier(PhaseBarrier pb);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    LogicalRegion logical_region;
    LogicalRegion parent_region;
    std::set<FieldID> fields;
  public:
    // This field is now optional (but required with control replication)
    PhysicalRegion physical_region;
  public:
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    Predicate predicate;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    bool silence_warnings;
  };

  /**
   * \struct ReleaseLauncher
   * A ReleaseLauncher supports the complementary operation to acquire
   * for performing user-level software coherence when dealing with
   * regions in simultaneous coherence mode.
   */
  struct ReleaseLauncher {
  public:
    ReleaseLauncher(
        LogicalRegion logical_region, LogicalRegion parent_region,
        PhysicalRegion physical_region = PhysicalRegion(),
        Predicate pred = Predicate::TRUE_PRED, MapperID id = 0,
        MappingTagID tag = 0, UntypedBuffer map_arg = UntypedBuffer(),
        const char* provenance = "");
  public:
    inline void add_field(FieldID f);
    inline void add_grant(Grant g);
    inline void add_wait_barrier(PhaseBarrier pb);
    inline void add_arrival_barrier(PhaseBarrier pb);
    inline void add_wait_handshake(LegionHandshake handshake);
    inline void add_arrival_handshake(LegionHandshake handshake);
  public:
    LogicalRegion logical_region;
    LogicalRegion parent_region;
    std::set<FieldID> fields;
  public:
    // This field is now optional (but required with control replication)
    PhysicalRegion physical_region;
  public:
    std::vector<Grant> grants;
    std::vector<PhaseBarrier> wait_barriers;
    std::vector<PhaseBarrier> arrive_barriers;
    Predicate predicate;
    MapperID map_id;
    MappingTagID tag;
    UntypedBuffer map_arg;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    // Inform the runtime about any static dependences
    // These will be ignored outside of static traces
    const std::vector<StaticDependence>* static_dependences;
  public:
    bool silence_warnings;
  };

  /**
   * \struct MustEpochLauncher
   * This is a meta-launcher object which contains other launchers.  The
   * purpose of this meta-launcher is to guarantee that all of the operations
   * specified in this launcher be guaranteed to run simultaneously.  This
   * enables the use of synchronization mechanisms such as phase barriers
   * and reservations between these operations without concern for deadlock.
   * If any condition is detected that will prevent simultaneous
   * parallel execution of the operations the runtime will report an error.
   * These conditions include true data dependences on regions as well
   * as cases where mapping decisions artificially serialize operations
   * such as two tasks being mapped to the same processor.
   */
  struct MustEpochLauncher {
  public:
    MustEpochLauncher(MapperID id = 0, MappingTagID tag = 0);
  public:
    inline void add_single_task(
        const DomainPoint& point, const TaskLauncher& launcher);
    inline void add_index_task(const IndexTaskLauncher& launcher);
  public:
    MapperID map_id;
    MappingTagID mapping_tag;
    std::vector<TaskLauncher> single_tasks;
    std::vector<IndexTaskLauncher> index_tasks;
  public:
    Domain launch_domain;
    IndexSpace launch_space;
    // Will only be used in control replication context. If left
    // unset the runtime will use launch_space/launch_domain
    IndexSpace sharding_space;
  public:
    // Provenance string for the runtime and tools to use
    std::string provenance;
  public:
    bool silence_warnings;
  };

}  // namespace Legion

#include "legion/api/launchers.inl"

#endif  // __LEGION_LAUNCHERS_H__
