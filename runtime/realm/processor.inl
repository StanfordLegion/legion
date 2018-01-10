/* Copyright 2018 Stanford University, NVIDIA Corporation
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

// processors for Realm

// nop, but helps IDEs
#include "realm/processor.h"

#include "realm/serialize.h"
TYPE_IS_SERIALIZABLE(Realm::Processor);
TYPE_IS_SERIALIZABLE(Realm::Processor::Kind);

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class Processor

  // use compiler-provided TLS for quickly finding our thread - stick this in another
  //  namespace to make it obvious
  namespace ThreadLocal {
    extern __thread Processor current_processor;
  };

  /*static*/ inline Processor Processor::get_executing_processor(void)
  { 
    return ThreadLocal::current_processor;
  }


}; // namespace Realm  
