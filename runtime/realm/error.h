/* Copyright 2023 Stanford University, NVIDIA Corporation
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

#include <errno.h>
#include <type_traits> // std::is_same

namespace Realm {

  template<typename T>
  struct ErrorHelper {
  public:
    static inline const char* process_error_message(T result, char *buffer)
    {
      // this is the version of strerror_r that returns an int so make
      // sure that it is not zero and then return the buffer
      assert(result == 0);
      return buffer;
    }
  };

  template<>
  struct ErrorHelper<char*> {
  public:
    static inline const char* process_error_message(char *result, char *buffer)
    {
      // this is the version of strerror_r that returns a string so use
      // that if it is not null
      return (result == NULL) ? buffer : result;
    }
  };

  const char* realm_strerror(int err, char *buffer, size_t size)
  {
    // Deal with the fact that strerror_r has two different possible
    // return types on different systems, call the right one based
    // on the return type and get the result
    // Return types should either be int or char*
    static_assert(
      std::is_same<decltype(strerror_r(err,buffer,size)),int>::value ||
      std::is_same<decltype(strerror_r(err,buffer,size)),char*>::value,
      "Unknown strerror_r return type");
    decltype(strerror_r(err,buffer,size)) result = strerror_r(err, buffer, size);

    return ErrorHelper<decltype(result)>::process_error_message(result, buffer);
  }

};
