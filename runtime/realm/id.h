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

// IDs (globally-usable handles) for Realm

#ifndef REALM_ID_H
#define REALM_ID_H

#include "realm/realm_c.h"
#include "realm/utils.h"

#include <iostream>

namespace Realm {

    class ID {
    public:
      typedef ::realm_id_t IDType;

      // Realm IDs are 64-bit values that uniquely encode both the type of the referred-to Realm
      //  object and its identity.  Different types use different fields and some need more bits
      //  than others, so even the type field is variable length.  The field order shown below is
      //  packed with the first field (the type tag) in the most significant bits of the 64-bit ID.

      // EVENT:       tag:1 = 0b1,  creator_node:16, gen_event_idx: 27, generation: 20
      // BARRIER:     tag:4 = 0x2,  creator_node:16, barrier_idx: 24, generation: 20
      // RESERVATION: tag:8 = 0x1f, creator_node:16, (unused):8, rsrv_idx: 32
      // MEMORY:      tag:8 = 0x1e, owner_node:16,   (unused):32, mem_idx: 8
      // IB_MEMORY:   tag:8 = 0x1a, owner_node:16,   (unused):32, mem_idx: 8
      // INSTANCE:    tag:2 = 0b01, owner_node:16,   creator_node:16, mem_idx: 8, inst_idx : 22
      // PROCESSOR:   tag:8 = 0x1d, owner_node:16,   (unused):28, proc_idx: 12
      // PROCGROUP:   tag:8 = 0x1c, owner_node:16,   creator_node:16, pgroup_idx: 24
      // SPARSITY:    tag:4 = 0x3,  owner_node:16,   creator_node:16, sparsity_idx: 28
      // COMPQUEUE:   tag:8 = 0x19, owner_node:16,   (unused):28, cq_idx: 12
      // SUBGRAPH:    tag:8 = 0x18, creator_node:16, (unused):16, subgraph_idx: 24

      static const int NODE_FIELD_WIDTH = 16;
      static const unsigned MAX_NODE_ID = (1U << NODE_FIELD_WIDTH) - 2; // reserve all 1's for special cases
      static const int EVENT_GENERATION_WIDTH = REALM_EVENT_GENERATION_BITS; // fom realm_c.h
      static const int MEMORY_INDEX_WIDTH = 8;
      static const int INSTANCE_INDEX_WIDTH = 22;

#define ACCESSOR(structname, name, field) \
      bitpack<IDType>::bitsliceref<structname::field> name ## _ ## field() { return id.slice<structname::field>(); } \
      bitpack<IDType>::constbitsliceref<structname::field> name ## _ ## field() const { return id.slice<structname::field>(); }

      struct FMT_Event {
	typedef bitfield<1, 63> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 63-NODE_FIELD_WIDTH> creator_node;
	typedef bitfield<63-NODE_FIELD_WIDTH-EVENT_GENERATION_WIDTH,
			 EVENT_GENERATION_WIDTH> gen_event_idx;
	typedef bitfield<EVENT_GENERATION_WIDTH, 0> generation;

	static const IDType TAG_VALUE = 1;
      };

      ACCESSOR(FMT_Event, event, type_tag)
      ACCESSOR(FMT_Event, event, creator_node)
      ACCESSOR(FMT_Event, event, gen_event_idx)
      ACCESSOR(FMT_Event, event, generation)


      struct FMT_Barrier {
	typedef bitfield<4, 60> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 60-NODE_FIELD_WIDTH> creator_node;
	typedef bitfield<24,
			 EVENT_GENERATION_WIDTH> barrier_idx;
	typedef bitfield<EVENT_GENERATION_WIDTH, 0> generation;  // MUST MATCH FMT_Event::generation size

	static const IDType TAG_VALUE = 2;
      };

      ACCESSOR(FMT_Barrier, barrier, type_tag)
      ACCESSOR(FMT_Barrier, barrier, creator_node)
      ACCESSOR(FMT_Barrier, barrier, barrier_idx)
      ACCESSOR(FMT_Barrier, barrier, generation)

      struct FMT_Reservation {
	typedef bitfield<8, 56> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 56-NODE_FIELD_WIDTH> creator_node;
	// middle bits unused
	typedef bitfield<32, 0> rsrv_idx;

	static const IDType TAG_VALUE = 0x1f;
      };

      ACCESSOR(FMT_Reservation, rsrv, type_tag)
      ACCESSOR(FMT_Reservation, rsrv, creator_node)
      ACCESSOR(FMT_Reservation, rsrv, rsrv_idx)

      struct FMT_Memory {
	typedef bitfield<8, 56> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 64-NODE_FIELD_WIDTH-MEMORY_INDEX_WIDTH> owner_node;
	// middle bits unused
	typedef bitfield<MEMORY_INDEX_WIDTH, 0> mem_idx;

	static const IDType TAG_VALUE = 0x1e;
      };

      ACCESSOR(FMT_Memory, memory, type_tag)
      ACCESSOR(FMT_Memory, memory, owner_node)
      ACCESSOR(FMT_Memory, memory, mem_idx)

      // IB memories use the same encoding as memories, but a different tag
      struct FMT_IB_Memory : public FMT_Memory {
        static const IDType TAG_VALUE = 0x1a;
      };

      struct FMT_Instance {
	typedef bitfield<2, 62> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 62-NODE_FIELD_WIDTH> owner_node;
	typedef bitfield<NODE_FIELD_WIDTH,
			 62-2*NODE_FIELD_WIDTH> creator_node;
	typedef bitfield<MEMORY_INDEX_WIDTH, INSTANCE_INDEX_WIDTH> mem_idx;
	typedef bitfield<INSTANCE_INDEX_WIDTH, 0> inst_idx;

	static const IDType TAG_VALUE = 1;
      };

      ACCESSOR(FMT_Instance, instance, type_tag)
      ACCESSOR(FMT_Instance, instance, owner_node)
      ACCESSOR(FMT_Instance, instance, creator_node)
      ACCESSOR(FMT_Instance, instance, mem_idx)
      ACCESSOR(FMT_Instance, instance, inst_idx)

      struct FMT_Processor {
	typedef bitfield<8, 56> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 56-NODE_FIELD_WIDTH> owner_node;
	// middle bits unused
	typedef bitfield<12, 0> proc_idx;

	static const IDType TAG_VALUE = 0x1d;
      };

      ACCESSOR(FMT_Processor, proc, type_tag)
      ACCESSOR(FMT_Processor, proc, owner_node)
      ACCESSOR(FMT_Processor, proc, proc_idx)

      struct FMT_ProcGroup {
	typedef bitfield<8, 56> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 56-NODE_FIELD_WIDTH> owner_node;
	typedef bitfield<NODE_FIELD_WIDTH,
			 56-2*NODE_FIELD_WIDTH> creator_node;
	typedef bitfield<24, 0> pgroup_idx;

	static const IDType TAG_VALUE = 0x1c;
      };

      ACCESSOR(FMT_ProcGroup, pgroup, type_tag)
      ACCESSOR(FMT_ProcGroup, pgroup, owner_node)
      ACCESSOR(FMT_ProcGroup, pgroup, creator_node)
      ACCESSOR(FMT_ProcGroup, pgroup, pgroup_idx)

      struct FMT_Sparsity {
	typedef bitfield<4, 60> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 60-NODE_FIELD_WIDTH> owner_node;
	typedef bitfield<NODE_FIELD_WIDTH,
			 60-2*NODE_FIELD_WIDTH> creator_node;
	typedef bitfield<28, 0> sparsity_idx;

	static const IDType TAG_VALUE = 0x3;
      };

      ACCESSOR(FMT_Sparsity, sparsity, type_tag)
      ACCESSOR(FMT_Sparsity, sparsity, owner_node)
      ACCESSOR(FMT_Sparsity, sparsity, creator_node)
      ACCESSOR(FMT_Sparsity, sparsity, sparsity_idx)

      struct FMT_CompQueue {
	typedef bitfield<8, 56> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 56-NODE_FIELD_WIDTH> owner_node;
	// middle bits unused
	typedef bitfield<12, 0> cq_idx;

	static const IDType TAG_VALUE = 0x19;
      };

      ACCESSOR(FMT_CompQueue, compqueue, type_tag)
      ACCESSOR(FMT_CompQueue, compqueue, owner_node)
      ACCESSOR(FMT_CompQueue, compqueue, cq_idx)

      struct FMT_Subgraph {
	typedef bitfield<8, 56> type_tag;
	typedef bitfield<NODE_FIELD_WIDTH,
			 56-NODE_FIELD_WIDTH> owner_node;
	typedef bitfield<NODE_FIELD_WIDTH,
			 56-2*NODE_FIELD_WIDTH> creator_node;
	typedef bitfield<24, 0> subgraph_idx;

	static const IDType TAG_VALUE = 0x18;
      };

      ACCESSOR(FMT_Subgraph, subgraph, type_tag)
      ACCESSOR(FMT_Subgraph, subgraph, owner_node)
      ACCESSOR(FMT_Subgraph, subgraph, creator_node)
      ACCESSOR(FMT_Subgraph, subgraph, subgraph_idx)

      static ID make_event(unsigned creator_node, unsigned gen_event_idx, unsigned generation);
      static ID make_barrier(unsigned creator_node, unsigned barrier_idx, unsigned generation);
      static ID make_reservation(unsigned creator_node, unsigned rsrv_idx);
      static ID make_memory(unsigned owner_node, unsigned mem_idx);
      static ID make_ib_memory(unsigned owner_node, unsigned mem_idx);
      static ID make_instance(unsigned owner_node, unsigned creator_node, unsigned mem_idx, unsigned inst_idx);
      static ID make_processor(unsigned owner_node, unsigned proc_idx);
      static ID make_procgroup(unsigned owner_node, unsigned creator_node, unsigned pgroup_idx);
      static ID make_sparsity(unsigned owner_node, unsigned creator_node, unsigned sparsity_idx);
      static ID make_compqueue(unsigned owner_node, unsigned cq_idx);
      static ID make_subgraph(unsigned owner_node, unsigned creator_node, unsigned subgraph_idx);

      bool is_null(void) const;
      bool is_event(void) const;
      bool is_barrier(void) const;
      bool is_reservation(void) const;
      bool is_memory(void) const;
      bool is_ib_memory(void) const;
      bool is_instance(void) const;
      bool is_processor(void) const;
      bool is_procgroup(void) const;
      bool is_sparsity(void) const;
      bool is_compqueue(void) const;
      bool is_subgraph(void) const;

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
	ID_SPARSITY,
	ID_SUBGRAPH,
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
      bool operator!=(const ID& rhs) const;

      template <class T>
      T convert(void) const;

      bitpack<IDType> id;

      friend std::ostream& operator<<(std::ostream& os, ID id);
    };

}; // namespace Realm

#include "realm/id.inl"

#endif // ifndef REALM_ID_H

