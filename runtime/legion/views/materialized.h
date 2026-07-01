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

#ifndef __LEGION_MATERIALIZED_VIEW_H__
#define __LEGION_MATERIALIZED_VIEW_H__

#include "legion/views/individual.h"
#include "legion/instances/physical.h"

namespace Legion {
  namespace Internal {

    /**
     * \class MaterializedView
     * This class represents a view on to a single normal physical
     * instance in a specific memory.
     */
    class MaterializedView
      : public IndividualView,
        public Heapify<MaterializedView, CONTEXT_LIFETIME> {
    public:
      struct DeferMaterializedViewArgs
        : public LgTaskArgs<DeferMaterializedViewArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_MATERIALIZED_VIEW_TASK_ID;
      public:
        DeferMaterializedViewArgs(void) = default;
        DeferMaterializedViewArgs(
            DistributedID d, PhysicalManager* m, AddressSpaceID log)
          : LgTaskArgs<DeferMaterializedViewArgs>(false, true), did(d),
            manager(m), logical_owner(log)
        { }
        void execute(void) const;
      public:
        DistributedID did;
        PhysicalManager* manager;
        AddressSpaceID logical_owner;
      };
    public:
      MaterializedView(
          DistributedID did, AddressSpaceID logical_owner,
          PhysicalManager* manager, bool register_now,
          CollectiveMapping* mapping = nullptr);
      MaterializedView(const MaterializedView& rhs) = delete;
      virtual ~MaterializedView(void);
    public:
      MaterializedView& operator=(const MaterializedView& rhs) = delete;
    public:
      inline const FieldMask& get_space_mask(void) const
      {
        return manager->layout->allocated_fields;
      }
    public:
      const FieldMask& get_physical_mask(void) const;
    public:
      virtual bool has_space(const FieldMask& space_mask) const;
    public:  // From InstanceView
      virtual void send_view(AddressSpaceID target) override;
    public:
      static void create_remote_view(
          DistributedID did, PhysicalManager* manager,
          AddressSpaceID logical_owner);
    protected:
      // Keep track of the current version numbers for each field
      // This will allow us to detect when physical instances are no
      // longer valid from a particular view when doing rollbacks for
      // resilience or mis-speculation.
      // typedef shrt::map<VersionID,FieldMaskMap<IndexSpaceExpression>,
      //                    PHYSICAL_VERSION_ALLOC> VersionFieldExprs;
      // VersionFieldExprs current_versions;
    };

    //--------------------------------------------------------------------------
    inline MaterializedView* LogicalView::as_materialized_view(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_materialized_view());
      return static_cast<MaterializedView*>(const_cast<LogicalView*>(this));
    }

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_MATERIALIZED_VIEW_H__
