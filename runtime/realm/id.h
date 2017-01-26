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

// IDs (globally-usable handles) for Realm

#ifndef REALM_ID_H
#define REALM_ID_H

#include <iostream>

namespace Realm {

    class ID {
    public:
      typedef unsigned long long IDType;

      // Realm IDs are 64-bit values that uniquely encode both the type of the referred-to Realm
      //  object and its identity.  Different types use different fields and some need more bits
      //  than others, so even the type field is variable length.  The field order shown below is
      //  packed with the first field (the type tag) in the most significant bits of the 64-bit ID.

      // EVENT:       tag:1 = 0b1,  creator_node:16, gen_event_idx: 27, generation: 20
      // BARRIER:     tag:4 = 0x7,  creator_node:16, barrier_idx: 24, generation: 20
      // RESERVATION: tag:8 = 0x1f, creator_node:16, (unused):8, rsrv_idx: 32
      // MEMORY:      tag:8 = 0x1e, owner_node:16,   (unused):28, mem_idx: 12
      // IB_MEMORY:   tag:8 = 0x1a, owner_node:16,   (unused):28, mem_idx: 12
      // INSTANCE:    tag:4 = 0x6,  owner_node:16,   creator_node:16, mem_idx: 12, inst_idx : 16
      // PROCESSOR:   tag:8 = 0x1d, owner_node:16,   (unused):28, proc_idx: 12
      // PROCGROUP:   tag:8 = 0x1c, owner_node:16,   creator_node:16, pgroup_idx: 24
      // IDXSPACE:    tag:4 = 0x5,  owner_node:16,   creator_node:16, idxspace_idx: 28
      // ALLOCATOR:   tag:8 = 0x1b, owner_node:16,   creator_node:16, allocator_idx: 24

      static const int NODE_FIELD_WIDTH = 16;
      static const unsigned MAX_NODE_ID = (1U << NODE_FIELD_WIDTH) - 2; // reserve all 1's for special cases

      struct FMT_Event {
	IDType generation : 20;
	IDType gen_event_idx : 27;
	IDType creator_node : NODE_FIELD_WIDTH;
	IDType type_tag : 1;
	static const IDType TAG_VALUE = 1;
      };

      struct FMT_Barrier {
	IDType generation : 20;  // MUST MATCH FMT_Event::generation size
	IDType barrier_idx : 24;
	IDType creator_node : 16;
	IDType type_tag : 4;
	static const IDType TAG_VALUE = 0x7;
      };

      struct FMT_Reservation {
	IDType rsrv_idx : 32;
	IDType unused : 8;
	IDType creator_node : 16;
	IDType type_tag : 8;
	static const IDType TAG_VALUE = 0x1f;
      };

      struct FMT_Memory {
	IDType mem_idx : 12;
	IDType unused : 28;
	IDType owner_node : 16;
	IDType type_tag : 8;
	static const IDType TAG_VALUE = 0x1e;
      };

      struct FMT_IB_Memory {
        IDType mem_idx : 12;
        IDType unused : 28;
        IDType owner_node : 16;
        IDType type_tag : 8;
        static const IDType TAG_VALUE = 0x1a;
      };

      struct FMT_Instance {
	IDType inst_idx : 16;
	IDType mem_idx : 12;
	IDType creator_node : 16;
	IDType owner_node : 16;
	IDType type_tag : 4;
	static const IDType TAG_VALUE = 0x6;
      };

      struct FMT_Processor {
	IDType proc_idx : 12;
	IDType unused : 28;
	IDType owner_node : 16;
	IDType type_tag : 8;
	static const IDType TAG_VALUE = 0x1d;
      };

      struct FMT_ProcGroup {
	IDType pgroup_idx : 24;
	IDType creator_node : 16;
	IDType owner_node : 16;
	IDType type_tag : 8;
	static const IDType TAG_VALUE = 0x1c;
      };

      struct FMT_IdxSpace {
	IDType idxspace_idx : 28;
	IDType creator_node : 16;
	IDType owner_node : 16;
	IDType type_tag : 4;
	static const IDType TAG_VALUE = 0x5;
      };

      struct FMT_Allocator {
	IDType allocator_idx : 24;
	IDType creator_node : 16;
	IDType owner_node : 16;
	IDType type_tag : 8;
	static const IDType TAG_VALUE = 0x1b;
      };

      static ID make_event(unsigned creator_node, unsigned gen_event_idx, unsigned generation);
      static ID make_barrier(unsigned creator_node, unsigned barrier_idx, unsigned generation);
      static ID make_reservation(unsigned creator_node, unsigned rsrv_idx);
      static ID make_memory(unsigned owner_node, unsigned mem_idx);
      static ID make_ib_memory(unsigned owner_node, unsigned mem_idx);
      static ID make_instance(unsigned owner_node, unsigned creator_node, unsigned mem_idx, unsigned inst_idx);
      static ID make_processor(unsigned owner_node, unsigned proc_idx);
      static ID make_procgroup(unsigned owner_node, unsigned creator_node, unsigned pgroup_idx);
      static ID make_idxspace(unsigned owner_node, unsigned creator_node, unsigned idxspace_idx);
      static ID make_allocator(unsigned owner_node, unsigned creator_node, unsigned allocator_idx);

      bool is_null(void) const;
      bool is_event(void) const;
      bool is_barrier(void) const;
      bool is_reservation(void) const;
      bool is_memory(void) const;
      bool is_ib_memory(void) const;
      bool is_instance(void) const;
      bool is_processor(void) const;
      bool is_procgroup(void) const;
      bool is_idxspace(void) const;
      bool is_allocator(void) const;

      enum ID_Types {
	ID_SPECIAL,
	ID_UNUSED_1,
	ID_EVENT,
	ID_BARRIER,
	ID_LOCK,
	ID_UNUSED_5,
	ID_MEMORY,
	ID_IB_MEMORY,
	ID_PROCESSOR,
	ID_PROCGROUP,
	ID_INDEXSPACE,
	ID_UNUSED_11,
	ID_ALLOCATOR,
	ID_UNUSED_13,
	ID_INSTANCE,
	ID_UNUSED_15,
      };

      static const IDType ID_NULL = 0;

      ID(void);
      ID(IDType _id);
      ID(const ID& copy_from);

      ID& operator=(const ID& copy_from);

      template <class T>
      ID(T thing_to_get_id_from);

      bool operator==(const ID& rhs) const;

      template <class T>
      T convert(void) const;

      union {
	IDType id;
	FMT_Event event;
	FMT_Barrier barrier;
	FMT_Reservation rsrv;
	FMT_Memory memory;
	FMT_IB_Memory ib_memory;
	FMT_Instance instance;
	FMT_Processor proc;
	FMT_ProcGroup pgroup;
	FMT_IdxSpace idxspace;
	FMT_Allocator allocator;
      };

      friend std::ostream& operator<<(std::ostream& os, ID id);
    };

}; // namespace Realm

#include "id.inl"

#endif // ifndef REALM_ID_H

