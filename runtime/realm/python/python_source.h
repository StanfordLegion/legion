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

#ifndef REALM_PYTHON_SOURCE_H
#define REALM_PYTHON_SOURCE_H

#include "realm/codedesc.h"

namespace Realm {

  class PythonSourceImplementation : public CodeImplementation {
  public:
    PythonSourceImplementation(const std::string& _module_name,
                               const std::vector<std::string>& _function_name);
    PythonSourceImplementation(const std::string& _module_name,
                               const std::string& _function_name);

    virtual ~PythonSourceImplementation(void);

    virtual CodeImplementation *clone(void) const;

    virtual bool is_portable(void) const;

    template <typename S>
    bool serialize(S& serializer) const;

    template <typename S>
    static CodeImplementation *deserialize_new(S& deserializer);

  protected:
    PythonSourceImplementation(void);

    static Serialization::PolymorphicSerdezSubclass<CodeImplementation, PythonSourceImplementation> serdez_subclass;

    virtual void print(std::ostream& os) const;

  public:
    std::string module_name;
    std::vector<std::string> function_name;
  };

}; // namespace Realm

#include "realm/python/python_source.inl"

#endif
