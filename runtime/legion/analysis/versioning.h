/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#ifndef __LEGION_VERSIONING_H__
#define __LEGION_VERSIONING_H__

#include "legion/analysis/equivalence_set.h"
#include "legion/utilities/dynamic_table.h"

namespace Legion {
  namespace Internal {

    /**
     * \class VersionInfo
     * A class for tracking version information about region usage
     */
    class VersionInfo : public Heapify<VersionInfo, OPERATION_LIFETIME> {
    public:
      VersionInfo(void);
      VersionInfo(const VersionInfo& rhs);
      virtual ~VersionInfo(void);
    public:
      VersionInfo& operator=(const VersionInfo& rhs);
    public:
      inline bool has_version_info(void) const
      {
        return !equivalence_sets.empty();
      }
      inline const op::FieldMaskMap<EquivalenceSet>& get_equivalence_sets(
          void) const
      {
        return equivalence_sets;
      }
      inline void swap(op::FieldMaskMap<EquivalenceSet>& others)
      {
        equivalence_sets.swap(others);
      }
      inline const FieldMask& get_valid_mask(void) const
      {
        return equivalence_sets.get_valid_mask();
      }
      inline void relax_valid_mask(const FieldMask& mask)
      {
        equivalence_sets.relax_valid_mask(mask);
      }
    public:
      void pack_equivalence_sets(Serializer& rez) const;
      void unpack_equivalence_sets(
          Deserializer& derez, std::set<RtEvent>& ready_events);
    public:
      void record_equivalence_set(EquivalenceSet* set, const FieldMask& mask);
      void clear(void);
    protected:
      op::FieldMaskMap<EquivalenceSet> equivalence_sets;
    };

    /**
     * \class VersionManager
     * The VersionManager class tracks the starting equivalence
     * sets for a given node in the logical region tree. Note
     * that its possible that these have since been shattered
     * and we need to traverse them, but it's a cached starting
     * point that doesn't involve tracing the entire tree.
     */
    class VersionManager : public EqSetTracker,
                           public Heapify<VersionManager, LONG_LIFETIME> {
    public:
      struct FinalizeOutputEquivalenceSetArgs
        : public LgTaskArgs<FinalizeOutputEquivalenceSetArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_FINALIZE_OUTPUT_EQ_SET_TASK_ID;
      public:
        FinalizeOutputEquivalenceSetArgs(void) = default;
        FinalizeOutputEquivalenceSetArgs(
            VersionManager* proxy, InnerContext* ctx, unsigned req_index,
            EquivalenceSet* s)
          : LgTaskArgs<FinalizeOutputEquivalenceSetArgs>(false, false),
            proxy_this(proxy), context(ctx), parent_req_index(req_index), set(s)
        {
          set->add_base_gc_ref(META_TASK_REF);
        }
        void execute(void) const;
      public:
        VersionManager* proxy_this;
        InnerContext* context;
        unsigned parent_req_index;
        EquivalenceSet* set;
      };
    public:
      VersionManager(RegionTreeNode* node, ContextID ctx);
      VersionManager(const VersionManager& manager) = delete;
      virtual ~VersionManager(void);
    public:
      VersionManager& operator=(const VersionManager& rhs) = delete;
    public:
      inline bool has_versions(const FieldMask& mask) const
      {
        return !(mask - equivalence_sets.get_valid_mask());
      }
      inline const FieldMask& get_version_mask(void) const
      {
        return equivalence_sets.get_valid_mask();
      }
    public:
      void perform_versioning_analysis(
          InnerContext* outermost, VersionInfo* version_info,
          RegionNode* region_node, const FieldMask& version_mask, Operation* op,
          unsigned index, unsigned parent_req_index, std::set<RtEvent>& ready,
          RtEvent* output_region_ready, bool collective_rendezvous);
      RtEvent finalize_output_equivalence_set(
          EquivalenceSet* set, InnerContext* enclosing,
          unsigned parent_req_index);
    public:
      virtual void add_subscription_reference(unsigned count = 1) override;
      virtual bool remove_subscription_reference(unsigned count = 1) override;
      virtual RegionTreeID get_region_tree_id(void) const override;
      virtual IndexSpaceExpression* get_tracker_expression(void) const override;
      virtual ReferenceSource get_reference_source_kind(void) const override
      {
        return VERSION_MANAGER_REF;
      }
    public:
      void finalize_manager(void);
    public:
      static void handle_finalize_output_eq_set(const void* args);
    public:
      const ContextID ctx;
      RegionTreeNode* const node;
    protected:
      mutable LocalLock manager_lock;
    };

    typedef DynamicTableAllocator<VersionManager, 10, 8>
        VersionManagerAllocator;

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_VERSIONING_H__
