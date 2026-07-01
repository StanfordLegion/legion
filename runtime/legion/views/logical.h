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

#ifndef __LEGION_LOGICAL_VIEW_H__
#define __LEGION_LOGICAL_VIEW_H__

#include "legion/kernel/garbage_collection.h"

namespace Legion {
  namespace Internal {

    /**
     * \class LogicalView
     * This class is the abstract base class for representing
     * the logical view onto one or more physical instances
     * in memory.  Logical views are reference counted
     * and will delete themselves once they no longer have
     * any valid handles.
     */
    class LogicalView : public DistributedCollectable {
    public:
      LogicalView(
          DistributedID did, bool register_now, CollectiveMapping* mapping);
      virtual ~LogicalView(void);
    public:
      inline bool is_instance_view(void) const;
      inline bool is_deferred_view(void) const;
      inline bool is_individual_view(void) const;
      inline bool is_collective_view(void) const;
      inline bool is_materialized_view(void) const;
      inline bool is_reduction_view(void) const;
      inline bool is_replicated_view(void) const;
      inline bool is_allreduce_view(void) const;
      inline bool is_fill_view(void) const;
      inline bool is_phi_view(void) const;
      inline bool is_reduction_kind(void) const;
    public:
      inline InstanceView* as_instance_view(void) const;
      inline DeferredView* as_deferred_view(void) const;
      inline IndividualView* as_individual_view(void) const;
      inline CollectiveView* as_collective_view(void) const;
      inline MaterializedView* as_materialized_view(void) const;
      inline ReductionView* as_reduction_view(void) const;
      inline ReplicatedView* as_replicated_view(void) const;
      inline AllreduceView* as_allreduce_view(void) const;
      inline FillView* as_fill_view(void) const;
      inline PhiView* as_phi_view(void) const;
    public:
      virtual void send_view(AddressSpaceID target) = 0;
    public:
      inline void add_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline void add_nested_valid_ref(DistributedID source, int cnt = 1);
      inline bool remove_base_valid_ref(ReferenceSource source, int cnt = 1);
      inline bool remove_nested_valid_ref(DistributedID source, int cnt = 1);
    public:
      virtual void pack_valid_ref(
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks) = 0;
      virtual void unpack_valid_ref(
          shrt::map<LogicalView*, LamportClock>& view_lamport_clocks,
          shrt::map<PhysicalManager*, LamportClock>& inst_lamport_clocks) = 0;
    protected:
#ifndef LEGION_DEBUG_GC
      void add_valid_reference(int cnt);
      bool remove_valid_reference(int cnt);
#else
      void add_base_valid_ref_internal(ReferenceSource source, int cnt);
      void add_nested_valid_ref_internal(DistributedID source, int cnt);
      bool remove_base_valid_ref_internal(ReferenceSource source, int cnt);
      bool remove_nested_valid_ref_internal(DistributedID source, int cnt);
#endif
      virtual void notify_valid(void) = 0;
      virtual bool notify_invalid(void) = 0;
    public:
      static inline DistributedID encode_materialized_did(DistributedID did);
      static inline DistributedID encode_reduction_did(DistributedID did);
      static inline DistributedID encode_replicated_did(DistributedID did);
      static inline DistributedID encode_allreduce_did(DistributedID did);
      static inline DistributedID encode_fill_did(DistributedID did);
      static inline DistributedID encode_phi_did(DistributedID did);
      static inline bool is_materialized_did(DistributedID did);
      static inline bool is_reduction_did(DistributedID did);
      static inline bool is_replicated_did(DistributedID did);
      static inline bool is_allreduce_did(DistributedID did);
      static inline bool is_individual_did(DistributedID did);
      static inline bool is_collective_did(DistributedID did);
      static inline bool is_fill_did(DistributedID did);
      static inline bool is_phi_did(DistributedID did);
    protected:
      mutable LocalLock view_lock;
    protected:
#ifdef LEGION_DEBUG_GC
      int valid_references;
#else
      std::atomic<int> valid_references;
#endif
#ifdef LEGION_DEBUG_GC
    protected:
      std::map<ReferenceSource, int> detailed_base_valid_references;
      std::map<DistributedID, int> detailed_nested_valid_references;
#endif
    };

  }  // namespace Internal
}  // namespace Legion

#include "legion/views/logical.inl"

#endif  // __LEGION_LOGICAL_VIEW_H__
