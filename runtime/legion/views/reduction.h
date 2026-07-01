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

#ifndef __LEGION_REDUCTION_VIEW_H__
#define __LEGION_REDUCTION_VIEW_H__

#include "legion/views/individual.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ReductionView
     * This class represents a single reduction physical instance
     * in a specific memory.
     */
    class ReductionView : public IndividualView,
                          public Heapify<ReductionView, CONTEXT_LIFETIME> {
    public:
      struct DeferReductionViewArgs
        : public LgTaskArgs<DeferReductionViewArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_REDUCTION_VIEW_TASK_ID;
      public:
        DeferReductionViewArgs(void) = default;
        DeferReductionViewArgs(
            DistributedID d, PhysicalManager* m, AddressSpaceID log)
          : LgTaskArgs<DeferReductionViewArgs>(false, true), did(d), manager(m),
            logical_owner(log)
        { }
        void execute(void) const;
      public:
        DistributedID did;
        PhysicalManager* manager;
        AddressSpaceID logical_owner;
      };
    public:
      ReductionView(
          DistributedID did, AddressSpaceID logical_owner,
          PhysicalManager* manager, bool register_now,
          CollectiveMapping* mapping = nullptr);
      ReductionView(const ReductionView& rhs) = delete;
      virtual ~ReductionView(void);
    public:
      ReductionView& operator=(const ReductionView& rhs) = delete;
    public:  // From InstanceView
      virtual void send_view(AddressSpaceID target) override;
      virtual ReductionOpID get_redop(void) const override;
      virtual FillView* get_redop_fill_view(void) const override
      {
        return fill_view;
      }
    public:
      static void create_remote_view(
          DistributedID did, PhysicalManager* manager,
          AddressSpaceID logical_owner);
    public:
      FillView* const fill_view;
    };

    //--------------------------------------------------------------------------
    inline ReductionView* LogicalView::as_reduction_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_reduction_view());
      return static_cast<ReductionView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_REDUCTION_VIEW_H__
