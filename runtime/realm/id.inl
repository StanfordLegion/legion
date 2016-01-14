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

// INCLDUED FROM id.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "id.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ID
  //

  inline ID::ID(IDType _value) : value(_value) {}

  template <class T>
  ID::ID(T thing_to_get_id_from) : value(thing_to_get_id_from.id) {}

  inline ID::ID(ID_Types _type, unsigned _node, IDType _index)
    : value((((IDType)_type) << (NODE_BITS + INDEX_BITS)) |
	    (((IDType)_node) << INDEX_BITS) |
	    _index)
  {}

  inline ID::ID(ID_Types _type, unsigned _node, IDType _index_h, IDType _index_l)
    : value((((IDType)_type) << (NODE_BITS + INDEX_BITS)) |
	    (((IDType)_node) << INDEX_BITS) |
	    (_index_h << INDEX_L_BITS) |
	    _index_l)
  {}

  inline bool ID::operator==(const ID& rhs) const 
  {
    return value == rhs.value;
  }

  inline ID::IDType ID::id(void) const
  {
    return value;
  }

  inline ID::ID_Types ID::type(void) const
  {
    return (ID_Types)(value >> (NODE_BITS + INDEX_BITS));
  }

  inline unsigned ID::node(void) const
  {
    return ((value >> INDEX_BITS) & ((1U << NODE_BITS)-1));
  }

  inline ID::IDType ID::index(void) const
  {
    return (value & ((((IDType)1) << INDEX_BITS) - 1));
  }
  
  inline ID::IDType ID::index_h(void) const
  {
    return ((value >> INDEX_L_BITS) & ((((IDType)1) << INDEX_H_BITS)-1));
  }

  inline ID::IDType ID::index_l(void) const
  {
    return (value & ((((IDType)1) << INDEX_L_BITS) - 1)); 
  }

  template <class T>
  T ID::convert(void) const { T thing_to_return; thing_to_return.id = value; return thing_to_return; }

  template <>
  inline ID ID::convert<ID>(void) const { return *this; }

}; // namespace Realm
