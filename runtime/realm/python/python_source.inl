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

namespace Realm {

  namespace Python {

    ////////////////////////////////////////////////////////////////////////
    //
    // class PythonSourceCodeImplementation

    inline void PythonSourceCodeImplementation::print(std::ostream& os) const
    {
      os << "pyref(" << module_name << "," << function_name << ")";
    }

    template <typename S>
    inline bool PythonSourceCodeImplementation::serialize(S& serializer) const
    {
      return (serializer << module_name) && (serializer << function_name);
    }

    template <typename S>
    inline /*static*/ CodeImplementation *PythonSourceCodeImplementation::deserialize_new(S& deserializer)
    {
      PythonSourceCodeImplementation *pyref = new PythonSourceCodeImplementation;
      if((deserializer >> pyref->module_name) && (deserializer >> pyref->function_name)) {
        return pyref;
      } else {
        delete pyref;
        return 0;
      }
    }

  }; // namespace Python

}; // namespace Realm
