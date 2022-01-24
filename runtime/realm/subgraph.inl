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

// nop, but helps IDEs
#include "realm/subgraph.h"

#include "realm/serialize.h"

TYPE_IS_SERIALIZABLE(Realm::Subgraph);

TYPE_IS_SERIALIZABLE(Realm::SubgraphDefinition::TaskDesc);
TYPE_IS_SERIALIZABLE(Realm::SubgraphDefinition::CopyDesc);
TYPE_IS_SERIALIZABLE(Realm::SubgraphDefinition::ArrivalDesc);
TYPE_IS_SERIALIZABLE(Realm::SubgraphDefinition::InstantiationDesc);
TYPE_IS_SERIALIZABLE(Realm::SubgraphDefinition::AcquireDesc);
TYPE_IS_SERIALIZABLE(Realm::SubgraphDefinition::ReleaseDesc);
TYPE_IS_SERIALIZABLE(Realm::SubgraphDefinition::Dependency);
TYPE_IS_SERIALIZABLE(Realm::SubgraphDefinition::Interpolation);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition

  inline SubgraphDefinition::SubgraphDefinition()
    : concurrency_mode(CONCURRENT)
  {}

  template <typename S>
  bool serdez(S& serdez, const SubgraphDefinition& s)
  {
    return ((serdez & s.tasks) &&
	    (serdez & s.copies) &&
	    (serdez & s.arrivals) &&
	    (serdez & s.instantiations) &&
	    (serdez & s.acquires) &&
	    (serdez & s.releases) &&
	    (serdez & s.dependencies) &&
	    (serdez & s.interpolations) &&
	    (serdez & s.concurrency_mode));
  }


  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition::TaskDesc

  inline SubgraphDefinition::TaskDesc::TaskDesc()
    : proc(Processor::NO_PROC)
    , task_id(static_cast<Processor::TaskFuncID>(-1))
    , priority(0)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition::CopyDesc

  inline SubgraphDefinition::CopyDesc::CopyDesc()
    : redop_id(0)
    , red_fold(false)
    , priority(0)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition::ArrivalDesc

  inline SubgraphDefinition::ArrivalDesc::ArrivalDesc()
    : barrier(Barrier::NO_BARRIER)
    , count(1)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition::InstantiationDesc

  inline SubgraphDefinition::InstantiationDesc::InstantiationDesc()
    : subgraph(Subgraph::NO_SUBGRAPH)
    , priority_adjust(0)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition::AcquireDesc

  inline SubgraphDefinition::AcquireDesc::AcquireDesc()
    : rsrv(Reservation::NO_RESERVATION)
    , mode(0)
    , exclusive(true)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition::ReleaseDesc

  inline SubgraphDefinition::ReleaseDesc::ReleaseDesc()
    : rsrv(Reservation::NO_RESERVATION)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition::Dependency

  inline SubgraphDefinition::Dependency::Dependency()
    : src_op_kind(OPKIND_INVALID)
    , src_op_index(0)
    , src_op_port(0)
    , tgt_op_kind(OPKIND_INVALID)
    , tgt_op_index(0)
    , tgt_op_port(0)
  {}


  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphDefinition::Interpolation

  inline SubgraphDefinition::Interpolation::Interpolation()
    : offset(0)
    , bytes(0)
    , target_kind(TARGET_INVALID)
    , target_index(0)
    , target_offset(0)
    , redop_id(0)
  {}


}; // namespace Realm
