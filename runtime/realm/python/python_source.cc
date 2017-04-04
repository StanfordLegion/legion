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

#include "python_source.h"

namespace Realm {

  namespace Python {

    ////////////////////////////////////////////////////////////////////////
    //
    // class PythonSourceCodeImplementation

    /*static*/ Serialization::PolymorphicSerdezSubclass<CodeImplementation,
                                                      PythonSourceCodeImplementation> PythonSourceCodeImplementation::serdez_subclass;

    PythonSourceCodeImplementation::PythonSourceCodeImplementation(void)
    {}

    PythonSourceCodeImplementation::PythonSourceCodeImplementation(const std::string& _module_name,
                                                                   const std::string& _function_name)
      : module_name(_module_name), function_name(_function_name)
    {}

    PythonSourceCodeImplementation::~PythonSourceCodeImplementation(void)
    {}

    CodeImplementation *PythonSourceCodeImplementation::clone(void) const
    {
      return new PythonSourceCodeImplementation(module_name, function_name);
    }

    bool PythonSourceCodeImplementation::is_portable(void) const
    {
      return true;
    }

  }; // namespace Python

}; // namespace Realm
