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

#ifndef __LEGION_REPLICATE_VIEW_H__
#define __LEGION_REPLICATE_VIEW_H__

#include "legion/views/collective.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ReplicatedView
     * This class represents a group of normal instances which all
     * must contain the same copy of data.
     */
    class ReplicatedView : public CollectiveView,
                           public Heapify<ReplicatedView, CONTEXT_LIFETIME> {
    public:
      ReplicatedView(
          DistributedID did, DistributedID ctx_did,
          const std::vector<IndividualView*>& views,
          const std::vector<DistributedID>& instances, bool register_now,
          CollectiveMapping* mapping);
      ReplicatedView(const ReplicatedView& rhs) = delete;
      virtual ~ReplicatedView(void);
    public:
      ReplicatedView& operator=(const ReplicatedView& rhs) = delete;
    public:  // From InstanceView
      virtual void send_view(AddressSpaceID target) override;
    };

    //--------------------------------------------------------------------------
    inline ReplicatedView* LogicalView::as_replicated_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_replicated_view());
      return static_cast<ReplicatedView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REPLICATE_VIEW_H__
