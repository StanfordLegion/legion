/* Copyright 2016 Stanford University, NVIDIA Corporation
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

namespace Realm {

    class ID {
    public:
#ifdef REALM_IDS_ARE_64BIT
      typedef unsigned long long IDType;

      enum {
	TYPE_BITS = 4,
	INDEX_H_BITS = 12,
	INDEX_L_BITS = 32,
	INDEX_BITS = INDEX_H_BITS + INDEX_L_BITS,
	NODE_BITS = 64 - TYPE_BITS - INDEX_BITS /* 16 = 64k nodes */
      };
#else
      // two forms of bit pack for IDs:
      //
      //  3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
      //  1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
      // +-----+---------------------------------------------------------+
      // |  TYP  |   NODE  |         INDEX                               |
      // |  TYP  |   NODE  |  INDEX_H    |           INDEX_L             |
      // +-----+---------------------------------------------------------+

      typedef unsigned int IDType;

      enum {
	TYPE_BITS = 4,
	INDEX_H_BITS = 7,
	INDEX_L_BITS = 16,
	INDEX_BITS = INDEX_H_BITS + INDEX_L_BITS,
	NODE_BITS = 32 - TYPE_BITS - INDEX_BITS /* 5 = 32 nodes */
      };
#endif

      enum ID_Types {
	ID_SPECIAL,
	ID_UNUSED_1,
	ID_EVENT,
	ID_BARRIER,
	ID_LOCK,
	ID_UNUSED_5,
	ID_MEMORY,
	ID_UNUSED_7,
	ID_PROCESSOR,
	ID_PROCGROUP,
	ID_INDEXSPACE,
	ID_SPARSITY,
	ID_ALLOCATOR,
	ID_UNUSED_13,
	ID_INSTANCE,
	ID_UNUSED_15,
      };

      enum ID_Specials {
	ID_INVALID = 0,
	ID_GLOBAL_MEM = (1U << INDEX_H_BITS) - 1,
      };

      ID(IDType _value);

      template <class T>
      ID(T thing_to_get_id_from);

      ID(ID_Types _type, unsigned _node, IDType _index);
      ID(ID_Types _type, unsigned _node, IDType _index_h, IDType _index_l);

      bool operator==(const ID& rhs) const;

      IDType id(void) const;
      ID_Types type(void) const;
      unsigned node(void) const;
      IDType index(void) const;
      IDType index_h(void) const;
      IDType index_l(void) const;

      template <class T>
      T convert(void) const;

    protected:
      IDType value;

      friend std::ostream& operator<<(std::ostream& os, ID id);
    };

    inline std::ostream& operator<<(std::ostream& os, ID id) { return os << std::hex << id.value << std::dec; }
	
}; // namespace Realm

#include "id.inl"

#endif // ifndef REALM_ID_H

