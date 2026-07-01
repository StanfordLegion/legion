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

#include "legion/api/values.h"
#include "legion/api/exception.h"
#include "legion/api/runtime.h"
#include "legion/contexts/context.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // UntypedDeferredValue
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  UntypedDeferredValue::UntypedDeferredValue(void)
    : instance(Realm::RegionInstance::NO_INST)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  UntypedDeferredValue::UntypedDeferredValue(
      size_t field_size, Memory memory, const void* initial_value,
      size_t alignment, bool escaping)
  //--------------------------------------------------------------------------
  {
    DeferredValueRequest request(
        memory, field_size, alignment, initial_value, escaping);
    Runtime* runtime = Runtime::get_runtime();
    *this = runtime->allocate_deferred_value(Runtime::get_context(), request);
  }

  //--------------------------------------------------------------------------
  UntypedDeferredValue::UntypedDeferredValue(
      size_t field_size, Memory::Kind memkind, const void* initial_value,
      size_t alignment, bool escaping)
  //--------------------------------------------------------------------------
  {
    DeferredValueRequest request(
        memkind, field_size, alignment, initial_value, escaping);
    Runtime* runtime = Runtime::get_runtime();
    *this = runtime->allocate_deferred_value(Runtime::get_context(), request);
  }

  //--------------------------------------------------------------------------
  UntypedDeferredValue::UntypedDeferredValue(const UntypedDeferredValue& rhs)
    : instance(rhs.instance)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  void UntypedDeferredValue::finalize(Context ctx) const
  //--------------------------------------------------------------------------
  {
    const size_t size = field_size();
    Runtime::legion_task_postamble(
        ctx, instance.pointer_untyped(0, size), size, true /*owner*/, instance);
  }

  //--------------------------------------------------------------------------
  Realm::RegionInstance UntypedDeferredValue::get_instance() const
  //--------------------------------------------------------------------------
  {
    return instance;
  }

  //--------------------------------------------------------------------------
  /*static*/ Domain UntypedDeferredValue::get_index_space_bounds(
      IndexSpace space)
  //--------------------------------------------------------------------------
  {
    return Runtime::get_runtime()->get_index_space_domain(space);
  }

  //--------------------------------------------------------------------------
  /*static*/ void UntypedDeferredValue::report_incompatible_accessor(
      const char* accessor_kind, bool buffer)
  //--------------------------------------------------------------------------
  {
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Incompatible " << accessor_kind << " for "
          << (buffer ? "(Untyped)DeferredBuffer" : "(Untyped)DeferredValue");
    error.raise();
  }

}  // namespace Legion
