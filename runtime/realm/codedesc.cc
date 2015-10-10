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

// constructs for describing code blobs to Realm

#include "codedesc.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Type

  std::ostream& operator<<(std::ostream& os, const Type& t)
  {
    switch(t.f_common.kind) {
    case Type::InvalidKind: os << "INVALIDTYPE"; break;
    case Type::OpaqueKind: os << "opaque(" << t.size_bits() << ")"; break;
    case Type::IntegerKind:
      {
	os << (t.f_integer.is_signed ? 's' : 'u') << "int(" << t.size_bits() << ")";
	break;
      }
    case Type::FloatingPointKind: os << "float(" << t.size_bits() << ")"; break;
    case Type::PointerKind:
      {
	os << *t.f_pointer.base_type;
	if(t.f_pointer.is_const) os << " const";
	os << " *";
	break;
      }
    case Type::FunctionPointerKind:
      {
	os << *t.f_funcptr.return_type << "(*)(";
	const std::vector<Type>& p = *t.f_funcptr.param_types;
	if(p.size()) {
	  for(size_t i = 0; i < p.size(); i++) {
	    if(i) os << ", ";
	    os << p[i];
	  }
	} else
	  os << "void";
	os << ")";
	break;
      }
    }
    return os;
  }

