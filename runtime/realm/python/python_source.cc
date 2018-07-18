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

#include "realm/python/python_source.h"

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class PythonSourceImplementation

  /*static*/ Serialization::PolymorphicSerdezSubclass<CodeImplementation,
                                                    PythonSourceImplementation> PythonSourceImplementation::serdez_subclass;

  PythonSourceImplementation::PythonSourceImplementation(void)
  {}

  PythonSourceImplementation::PythonSourceImplementation(const std::string& _module_name,
                                                         const std::vector<std::string>& _function_name)
    : module_name(_module_name), function_name(_function_name)
  {}

  PythonSourceImplementation::PythonSourceImplementation(const std::string& _module_name,
                                                         const std::string& _function_name)
    : module_name(_module_name), function_name(1, _function_name)
  {}

  PythonSourceImplementation::~PythonSourceImplementation(void)
  {}

  CodeImplementation *PythonSourceImplementation::clone(void) const
  {
    return new PythonSourceImplementation(module_name, function_name);
  }

  bool PythonSourceImplementation::is_portable(void) const
  {
    return true;
  }

}; // namespace Realm
