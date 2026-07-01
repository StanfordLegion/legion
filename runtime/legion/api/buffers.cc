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
  // UntypedDeferredBuffer
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  template<typename T>
  /*static*/ UntypedDeferredBuffer<T> UntypedDeferredBuffer<T>::allocate_buffer(
      const DeferredBufferRequest& request)
  //--------------------------------------------------------------------------
  {
    Runtime* runtime = Runtime::get_runtime();
    return runtime->allocate_deferred_buffer<T>(
        Runtime::get_context(), request);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  void UntypedDeferredBuffer<T>::destroy(Realm::Event precondition)
  //--------------------------------------------------------------------------
  {
    Runtime* runtime = Runtime::get_runtime();
    runtime->destroy_deferred_buffer<T>(
        Runtime::get_context(), *this, precondition);
  }

  //--------------------------------------------------------------------------
  template<typename T>
  /*static*/ void UntypedDeferredBuffer<T>::report_nondense_rect(void)
  //--------------------------------------------------------------------------
  {
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Illegal request for point of non-dense rectangle in a "
          << "DeferredBuffer. Make sure that you only ask for rectangles "
          << "that are dense in the layout of the deferred buffer or use "
          << "the version that passes back strides.";
    error.raise();
  }

  // Instantiate this for all the coordinate types that Realm supports
  template class UntypedDeferredBuffer<int>;
  template class UntypedDeferredBuffer<unsigned>;
  template class UntypedDeferredBuffer<long long>;

}  // namespace Legion
