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

#include "legion/views/deferred.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // DeferredView
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    DeferredView::DeferredView(
        DistributedID did, bool register_now, CollectiveMapping* mapping)
      : LogicalView(did, register_now, mapping)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    DeferredView::~DeferredView(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void DeferredView::notify_valid(void)
    //--------------------------------------------------------------------------
    {
      add_base_gc_ref(INTERNAL_VALID_REF);
    }

    //--------------------------------------------------------------------------
    bool DeferredView::notify_invalid(void)
    //--------------------------------------------------------------------------
    {
      return remove_base_gc_ref(INTERNAL_VALID_REF);
    }

  }  // namespace Internal
}  // namespace Legion
