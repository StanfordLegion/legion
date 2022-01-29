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

#ifndef LLVMJIT_H
#define LLVMJIT_H

// NOTE: let's try to keep LLVM include files/etc. out of here, because they bring
//  a bunch of C++11 baggage

#include "realm/realm_config.h"
#include "realm/codedesc.h"
#include "realm/bytearray.h"

#include <string>

namespace Realm {

#ifdef REALM_ALLOW_MISSING_LLVM_LIBS
  namespace LLVMJit {
    extern bool llvmjit_available;
  };
#endif

  // a code implementation that handles LLVM IR - text only right now, but the plan
  //  is to support binary here too

  class REALM_PUBLIC_API LLVMIRImplementation : public CodeImplementation {
  public:
    LLVMIRImplementation(const void *irdata, size_t irlen, const std::string& _entry_symbol);

    virtual ~LLVMIRImplementation(void);

    virtual CodeImplementation *clone(void) const;

    virtual bool is_portable(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static CodeImplementation *deserialize_new(S& deserializer);

  protected:
    LLVMIRImplementation(void);

    static Serialization::PolymorphicSerdezSubclass<CodeImplementation, LLVMIRImplementation> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    ByteArray ir;
    std::string entry_symbol;
  };

}; // namespace Realm

#include "realm/llvmjit/llvmjit.inl"

#endif
