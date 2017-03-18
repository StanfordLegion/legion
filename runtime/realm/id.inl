/* Copyright 2017 Stanford University, NVIDIA Corporation
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
#include "id.h"

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
    id.event.type_tag = FMT_Event::TAG_VALUE;
    id.event.creator_node = creator_node;
    id.event.gen_event_idx = gen_event_idx;
    id.event.generation = generation;
    return id;
  }

  inline bool ID::is_event(void) const
  {
    return this->event.type_tag == FMT_Event::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_barrier(unsigned creator_node, unsigned barrier_idx, unsigned generation)
  {
    ID id;
    id.barrier.type_tag = FMT_Barrier::TAG_VALUE;
    id.barrier.creator_node = creator_node;
    id.barrier.barrier_idx = barrier_idx;
    id.barrier.generation = generation;
    return id;
  }

  inline bool ID::is_barrier(void) const
  {
    return this->barrier.type_tag == FMT_Barrier::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_reservation(unsigned creator_node, unsigned rsrv_idx)
  {
    ID id;
    id.rsrv.type_tag = FMT_Reservation::TAG_VALUE;
    id.rsrv.creator_node = creator_node;
    id.rsrv.unused = 0;
    id.rsrv.rsrv_idx = rsrv_idx;
    return id;
  }

  inline bool ID::is_reservation(void) const
  {
    return this->rsrv.type_tag == FMT_Reservation::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_memory(unsigned owner_node, unsigned mem_idx)
  {
    ID id;
    id.memory.type_tag = FMT_Memory::TAG_VALUE;
    id.memory.owner_node = owner_node;
    id.memory.unused = 0;
    id.memory.mem_idx = mem_idx;
    return id;
  }

  inline bool ID::is_memory(void) const
  {
    return this->memory.type_tag == FMT_Memory::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_ib_memory(unsigned owner_node, unsigned mem_idx)
  {
    ID id;
    id.memory.type_tag = FMT_IB_Memory::TAG_VALUE;
    id.memory.owner_node = owner_node;
    id.memory.unused = 0;
    id.memory.mem_idx = mem_idx;
    return id;
  }

  inline bool ID::is_ib_memory(void) const
  {
    return this->memory.type_tag == FMT_IB_Memory::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_instance(unsigned owner_node, unsigned creator_node, unsigned mem_idx, unsigned inst_idx)
  {
    ID id;
    id.instance.type_tag = FMT_Instance::TAG_VALUE;
    id.instance.owner_node = owner_node;
    id.instance.creator_node = creator_node;
    id.instance.mem_idx = mem_idx;
    id.instance.inst_idx = inst_idx;
    return id;
  }

  inline bool ID::is_instance(void) const
  {
    return this->instance.type_tag == FMT_Instance::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_processor(unsigned owner_node, unsigned proc_idx)
  {
    ID id;
    id.proc.type_tag = FMT_Processor::TAG_VALUE;
    id.proc.owner_node = owner_node;
    id.proc.unused = 0;
    id.proc.proc_idx = proc_idx;
    return id;
  }

  inline bool ID::is_processor(void) const
  {
    return this->proc.type_tag == FMT_Processor::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_procgroup(unsigned owner_node, unsigned creator_node, unsigned pgroup_idx)
  {
    ID id;
    id.pgroup.type_tag = FMT_ProcGroup::TAG_VALUE;
    id.pgroup.owner_node = owner_node;
    id.pgroup.creator_node = creator_node;
    id.pgroup.pgroup_idx = pgroup_idx;
    return id;
  }

  inline bool ID::is_procgroup(void) const
  {
    return this->pgroup.type_tag == FMT_ProcGroup::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_idxspace(unsigned owner_node, unsigned creator_node, unsigned idxspace_idx)
  {
    ID id;
    id.idxspace.type_tag = FMT_IdxSpace::TAG_VALUE;
    id.idxspace.owner_node = owner_node;
    id.idxspace.creator_node = creator_node;
    id.idxspace.idxspace_idx = idxspace_idx;
    return id;
  }

  inline bool ID::is_idxspace(void) const
  {
    return this->idxspace.type_tag == FMT_IdxSpace::TAG_VALUE;
  }

  /*static*/ inline ID ID::make_allocator(unsigned owner_node, unsigned creator_node, unsigned allocator_idx)
  {
    ID id;
    id.allocator.type_tag = FMT_Allocator::TAG_VALUE;
    id.allocator.owner_node = owner_node;
    id.allocator.creator_node = creator_node;
    id.allocator.allocator_idx = allocator_idx;
    return id;
  }

  inline bool ID::is_allocator(void) const
  {
    return this->allocator.type_tag == FMT_Allocator::TAG_VALUE;
  }

  inline std::ostream& operator<<(std::ostream& os, ID id) { return os << std::hex << id.id << std::dec; }

}; // namespace Realm
