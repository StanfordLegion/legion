/* Copyright 2026 Stanford University, NVIDIA Corporation
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

// Included from future.h - do not include this directly

// Useful for IDEs
#include "legion/api/future.h"

namespace Legion {

  //--------------------------------------------------------------------------
  template<typename T, PrivilegeMode PM>
  inline Span<T, PM> Future::get_span(
      Memory::Kind memory, bool silence_warnings,
      const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    // This has to be true for now
    static_assert(
        PM == LEGION_READ_ONLY,
        "PrivilegeMode for Future:get_span must be 'LEGION_READ_ONLY' "
        "currently");
    size_t size = 0;
    const void* ptr = get_buffer(
        memory, &size, false /*check size*/, silence_warnings, warning_string);
    legion_assert((size % sizeof(T)) == 0);
    return Span<T, PM>(ptr, size / sizeof(T));
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline const T& Future::get_reference(
      bool silence_warnings, const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    size_t size = sizeof(T);
    const void* ptr = get_buffer(
        Memory::SYSTEM_MEM, &size, true /*check size*/, silence_warnings,
        warning_string);
    legion_assert(size == sizeof(T));
    return *static_cast<const T*>(ptr);
  }

  //--------------------------------------------------------------------------
  inline const void* Future::get_untyped_pointer(
      bool silence_warnings, const char* warning_string) const
  //--------------------------------------------------------------------------
  {
    size_t size = 0;
    return get_buffer(
        Memory::SYSTEM_MEM, &size, false /*check size*/, silence_warnings,
        warning_string);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  inline T Future::get(void)
  //--------------------------------------------------------------------------
  {
    return get_result<T>();
  }

  //--------------------------------------------------------------------------
  inline bool Future::valid(void) const
  //--------------------------------------------------------------------------
  {
    return (impl != nullptr);
  }

  //--------------------------------------------------------------------------
  inline void Future::wait(void) const
  //--------------------------------------------------------------------------
  {
    get_void_result();
  }

}  // namespace Legion

namespace std {

#define LEGION_DEFINE_HASHABLE(__TYPE_NAME__)                     \
  template<>                                                      \
  struct hash<__TYPE_NAME__> {                                    \
    inline std::size_t operator()(const __TYPE_NAME__& obj) const \
    {                                                             \
      return obj.hash();                                          \
    }                                                             \
  };

  LEGION_DEFINE_HASHABLE(Legion::Future);

#undef LEGION_DEFINE_HASHABLE

};  // namespace std
