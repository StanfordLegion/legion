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

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class PythonSourceImplementation

  inline void PythonSourceImplementation::print(std::ostream& os) const
  {
    os << "pyref(" << module_name;
    for (std::vector<std::string>::const_iterator it = function_name.begin(),
           ie = function_name.end(); it != ie; ++it) {
      os << "." << *it;
    }
    os << ")";
  }

  template <typename S>
  inline bool PythonSourceImplementation::serialize(S& serializer) const
  {
    return (serializer << module_name) && (serializer << function_name);
  }

  template <typename S>
  inline /*static*/ CodeImplementation *PythonSourceImplementation::deserialize_new(S& deserializer)
  {
    PythonSourceImplementation *pyref = new PythonSourceImplementation;
    if((deserializer >> pyref->module_name) && (deserializer >> pyref->function_name)) {
      return pyref;
    } else {
      delete pyref;
      return 0;
    }
  }

}; // namespace Realm
