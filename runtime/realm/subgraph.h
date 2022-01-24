/* Copyright 2022 Stanford University, NVIDIA Corporation
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

// subgraphs for Realm

#ifndef REALM_SUBGRAPH_H
#define REALM_SUBGRAPH_H

#include "realm/realm_c.h"

#include "realm/event.h"
#include "realm/indexspace.h"
#include "realm/processor.h"
#include "realm/profiling.h"
#include "realm/reservation.h"

namespace Realm {

  // TODO: C equivalent: realm_subgraph_defn_t
  struct SubgraphDefinition;

  class REALM_PUBLIC_API Subgraph {
  public:
    typedef ::realm_id_t id_t;

    id_t id;
    bool operator<(const Subgraph& rhs) const { return id < rhs.id; }
    bool operator==(const Subgraph& rhs) const { return id == rhs.id; }
    bool operator!=(const Subgraph& rhs) const { return id != rhs.id; }

    static const Subgraph NO_SUBGRAPH;

    bool exists(void) const { return id != 0; }

    static Event create_subgraph(Subgraph& subgraph,
				 const SubgraphDefinition& defn,
				 const ProfilingRequestSet& prs,
				 Event wait_on = Event::NO_EVENT);

    // TODO: collective construction

    void destroy(Event wait_on = Event::NO_EVENT) const;

    Event instantiate(const void *args, size_t arglen,
		      const ProfilingRequestSet& prs,
		      Event wait_on = Event::NO_EVENT,
		      int priority_adjust = 0) const;

    Event instantiate(const void *args, size_t arglen,
		      const ProfilingRequestSet& prs,
		      const std::vector<Event>& preconditions,
		      std::vector<Event>& postconditions,
		      Event wait_on = Event::NO_EVENT,
		      int priority_adjust = 0) const;

    // TODO: collective instantiation
  };

  struct REALM_PUBLIC_API SubgraphDefinition {
    SubgraphDefinition();

    // operations permitted in a subgraph:
    //  task spawn
    //  copy/fill
    //  barrier arrival
    //  subgraph instantiation
    //  reservation acquire/release

    // operations not permitted in a subgraph:
    //  instance creation/destruction
    //  subgraph creation/destruction
    //  reservation creation/destruction
    //  barrier creation/destruction
    //  user event creation/triggering
    //  barrier wait (use a precondition)
    //  task registration

    struct TaskDesc {
      TaskDesc();  // initializes all fields

      // interpolatable: args
      Processor proc;
      Processor::TaskFuncID task_id;
      ByteArray args;
      int priority /*= 0*/;
      ProfilingRequestSet prs;
    };

    struct CopyDesc {
      CopyDesc();  // initializes all fields

      // interpolatable: none?
      IndexSpaceGeneric space; // type-erase here to avoid template explosion
      std::vector<CopySrcDstField> srcs;
      std::vector<CopySrcDstField> dsts;
      ProfilingRequestSet prs;
      ReductionOpID redop_id /*= 0*/;
      bool red_fold /*= false*/;
      int priority /*= 0*/;
    };

    struct ArrivalDesc {
      ArrivalDesc();  // initializes all fields
      
      // interpolatable: { barrier, reduce_value }
      Barrier barrier;
      unsigned count /*= 1*/;
      ByteArray reduce_value;
    };

    struct InstantiationDesc {
      InstantiationDesc();  // initializes all fields

      // interpolatable: args
      Subgraph subgraph;
      ByteArray args;
      ProfilingRequestSet prs;
      int priority_adjust;
    };

    struct AcquireDesc {
      AcquireDesc();  // initializes all fields

      // interpolatable: none?
      Reservation rsrv;
      unsigned mode /*= 0*/;
      bool exclusive /*= true*/;
    };

    struct ReleaseDesc {
      ReleaseDesc();  // initializes all fields

      // interpolatable: none?
      Reservation rsrv;
    };

    std::vector<TaskDesc> tasks;
    std::vector<CopyDesc> copies;
    std::vector<ArrivalDesc> arrivals;
    std::vector<InstantiationDesc> instantiations;
    std::vector<AcquireDesc> acquires;
    std::vector<ReleaseDesc> releases;

    // dependencies between subgraph operations and/or external
    //  pre/post-conditions
    // operations are defined by their kind and their index
    // most operations have exactly one input "port" and output "port", but:
    //  arrivals and releases have zero outputs
    //  an instantiation has as many ports as it was defined with (including
    //   the "default port" 0)
    // the same target op/port may be subject to multiple dependencies - the
    //  equivalent of event merging happens behind the scenes

    enum OpKind {
      OPKIND_INVALID,
      OPKIND_TASK,
      OPKIND_COPY,
      OPKIND_ARRIVAL,
      OPKIND_INSTANTIATION,
      OPKIND_ACQUIRE,
      OPKIND_RELEASE,
      OPKIND_EXT_PRECOND,
      OPKIND_EXT_POSTCOND,
      OPKIND_COLL_PRECOND,  // these are used to connect subgraph contributions
      OPKIND_COLL_POSTCOND, //  from different ranks in collective construction
    };

    struct Dependency {
      Dependency();  // initializes all fields

      OpKind src_op_kind;
      unsigned src_op_index;
      unsigned src_op_port;
      OpKind tgt_op_kind;
      unsigned tgt_op_index;
      unsigned tgt_op_port;
    };

    std::vector<Dependency> dependencies;

    // interpolation allows values provided at subgraph instantiation time to
    //  overwrite/combine with argument values provided at subgraph creation
    //  time
    // TODO: could use this to allow looking at barrier values?
    // TODO: allow _all_ operation parameters to be interpolated, even if
    //   it means subgraph cannot be pre-compiled?

    struct Interpolation {
      Interpolation();  // initializes all fields

      enum TargetKind {
	TARGET_INVALID,

	TARGET_TASK_BASE = OPKIND_TASK << 8,
	TARGET_TASK_ARGS,

	TARGET_COPY_BASE = OPKIND_COPY << 8,

	TARGET_ARRIVAL_BASE = OPKIND_ARRIVAL << 8,
	TARGET_ARRIVAL_BARRIER,
	TARGET_ARRIVAL_VALUE,

	TARGET_INSTANCE_BASE = OPKIND_INSTANTIATION << 8,
	TARGET_INSTANCE_ARGS,

	TARGET_ACQUIRE_BASE = OPKIND_ACQUIRE << 8,

	TARGET_RELEASE_BASE = OPKIND_RELEASE << 8,
      };

      size_t offset;           // offset within instantiation args
      size_t bytes;            // size within instantiation args
      TargetKind target_kind;
      unsigned target_index;   // index of operation within its list
      size_t target_offset;    // offset within target field
      ReductionOpID redop_id;  // overwrite if 0, reduce apply otherwise
    };

    std::vector<Interpolation> interpolations;

    // concurrency mode - more serial versions may allow the compiled form to
    //  pre-allocate and reuse resources, improving efficiency
    enum ConcurrencyMode {
      ONE_SHOT,             // can only be executed once (e.g. probably not
                            //   worth optimizing)
      INSTANTIATION_ORDER,  // implementation may serialize instantiations in
                            //   in the order they were instantiated
      SERIALIZABLE,         // implementations may be serialized, but a later
                            //   instatiation must be able to run even if
                            //   earlier instantiations are waiting for
                            //   satisfaction of their default precondition
      CONCURRENT,           // concurrent execution of instantiations is
                            //   required for correctness
    };

    ConcurrencyMode concurrency_mode;

    // longer term possibilites:
    //  conditional execution
    //  loops
    //  local "scratchpad" for small-value-communication
  };

}; // namespace Realm

#include "realm/subgraph.inl"

#endif // ifndef REALM_SUBGRAPH_H
