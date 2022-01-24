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

// INCLDUED FROM id.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "realm/id.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ID
  //

  inline ID::ID(IDType _id) : id(_id) {}

  template <class T>
  ID::ID(T thing_to_get_id_from) : id(thing_to_get_id_from.id) {}

  inline ID::ID(void) : id(0) {}

  inline ID::ID(const ID& copy_from) : id(copy_from.id) {}

  inline ID& ID::operator=(const ID& copy_from)
  {
    id = copy_from.id;
    return *this;
  }

  inline bool ID::operator==(const ID& rhs) const 
  {
    return this->id == rhs.id;
  }

  inline bool ID::operator!=(const ID& rhs) const 
  {
    return this->id != rhs.id;
  }

  template <class T>
  T ID::convert(void) const { T thing_to_return; thing_to_return.id = id; return thing_to_return; }

  template <>
  inline ID ID::convert<ID>(void) const { return *this; }

  inline bool ID::is_null(void) const
  {
    return this->id == 0;
  }

  /*static*/ inline ID ID::make_event(unsigned creator_node, unsigned gen_event_idx, unsigned generation)
  {
    ID id;
    id.id = 0;
    id.event_type_tag() |= FMT_Event::TAG_VALUE;
    id.event_creator_node() |= creator_node;
    id.event_gen_event_idx() |= gen_event_idx;
    id.event_generation() |= generation;
    return id;
  }

  inline bool ID::is_event(void) const
  {
    return this->event_type_tag() == FMT_Event::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_barrier(unsigned creator_node, unsigned barrier_idx, unsigned generation)
  {
    ID id;
    id.id = 0;
    id.barrier_type_tag() |= FMT_Barrier::TAG_VALUE;
    id.barrier_creator_node() |= creator_node;
    id.barrier_barrier_idx() |= barrier_idx;
    id.barrier_generation() |= generation;
    return id;
  }

  inline bool ID::is_barrier(void) const
  {
    return this->barrier_type_tag() == FMT_Barrier::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_reservation(unsigned creator_node, unsigned rsrv_idx)
  {
    ID id;
    id.id = 0;
    id.rsrv_type_tag() |= FMT_Reservation::TAG_VALUE;
    id.rsrv_creator_node() |= creator_node;
    id.rsrv_rsrv_idx() |= rsrv_idx;
    return id;
  }

  inline bool ID::is_reservation(void) const
  {
    return this->rsrv_type_tag() == FMT_Reservation::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_memory(unsigned owner_node, unsigned mem_idx)
  {
    ID id;
    id.id = 0;
    id.memory_type_tag() |= FMT_Memory::TAG_VALUE;
    id.memory_owner_node() |= owner_node;
    id.memory_mem_idx() |= mem_idx;
    return id;
  }

  inline bool ID::is_memory(void) const
  {
    return this->memory_type_tag() == FMT_Memory::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_ib_memory(unsigned owner_node, unsigned mem_idx)
  {
    ID id;
    id.id = 0;
    id.memory_type_tag() |= FMT_IB_Memory::TAG_VALUE;
    id.memory_owner_node() |= owner_node;
    id.memory_mem_idx() |= mem_idx;
    return id;
  }

  inline bool ID::is_ib_memory(void) const
  {
    return this->memory_type_tag() == FMT_IB_Memory::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_instance(unsigned owner_node, unsigned creator_node, unsigned mem_idx, unsigned inst_idx)
  {
    ID id;
    id.id = 0;
    id.instance_type_tag() |= FMT_Instance::TAG_VALUE;
    id.instance_owner_node() |= owner_node;
    id.instance_creator_node() |= creator_node;
    id.instance_mem_idx() |= mem_idx;
    id.instance_inst_idx() |= inst_idx;
    return id;
  }

  inline bool ID::is_instance(void) const
  {
    return this->instance_type_tag() == FMT_Instance::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_processor(unsigned owner_node, unsigned proc_idx)
  {
    ID id;
    id.id = 0;
    id.proc_type_tag() |= FMT_Processor::TAG_VALUE;
    id.proc_owner_node() |= owner_node;
    id.proc_proc_idx() |= proc_idx;
    return id;
  }

  inline bool ID::is_processor(void) const
  {
    return this->proc_type_tag() == FMT_Processor::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_procgroup(unsigned owner_node, unsigned creator_node, unsigned pgroup_idx)
  {
    ID id;
    id.id = 0;
    id.pgroup_type_tag() |= FMT_ProcGroup::TAG_VALUE;
    id.pgroup_owner_node() |= owner_node;
    id.pgroup_creator_node() |= creator_node;
    id.pgroup_pgroup_idx() |= pgroup_idx;
    return id;
  }

  inline bool ID::is_procgroup(void) const
  {
    return this->pgroup_type_tag() == FMT_ProcGroup::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_sparsity(unsigned owner_node, unsigned creator_node, unsigned sparsity_idx)
  {
    ID id;
    id.id = 0;
    id.sparsity_type_tag() |= FMT_Sparsity::TAG_VALUE;
    id.sparsity_owner_node() |= owner_node;
    id.sparsity_creator_node() |= creator_node;
    id.sparsity_sparsity_idx() |= sparsity_idx;
    return id;
  }

  inline bool ID::is_sparsity(void) const
  {
    return this->sparsity_type_tag() == FMT_Sparsity::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_compqueue(unsigned owner_node, unsigned cq_idx)
  {
    ID id;
    id.id = 0;
    id.compqueue_type_tag() |= FMT_CompQueue::TAG_VALUE;
    id.compqueue_owner_node() |= owner_node;
    id.compqueue_cq_idx() |= cq_idx;
    return id;
  }

  inline bool ID::is_compqueue(void) const
  {
    return this->compqueue_type_tag() == FMT_CompQueue::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_subgraph(unsigned owner_node, unsigned creator_node, unsigned subgraph_idx)
  {
    ID id;
    id.id = 0;
    id.subgraph_type_tag() |= FMT_Subgraph::TAG_VALUE;
    id.subgraph_owner_node() |= owner_node;
    id.subgraph_creator_node() |= creator_node;
    id.subgraph_subgraph_idx() |= subgraph_idx;
    return id;
  }

  inline bool ID::is_subgraph(void) const
  {
    return this->subgraph_type_tag() == FMT_Subgraph::TAG_VALUE;
  }

  inline std::ostream& operator<<(std::ostream& os, ID id) { return os << std::hex << static_cast<ID::IDType>(id.id) << std::dec; }

}; // namespace Realm
