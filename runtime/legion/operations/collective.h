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

#ifndef __LEGION_COLLECTIVE_OPERATION_H__
#define __LEGION_COLLECTIVE_OPERATION_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/operations/operation.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * class CollectiveHelperOp
     * This is a small class that helps behave like an operation
     * for the other types that might want to perform collective
     * rendezvous but are not an operation like a ShardManager
     */
    class CollectiveHelperOp : public DistributedCollectable {
    public:
      CollectiveHelperOp(
          DistributedID did, bool register_with_runtime = true,
          CollectiveMapping* mapping = nullptr)
        : DistributedCollectable(did, register_with_runtime, mapping)
      { }
    public:
      virtual InnerContext* get_context(void) = 0;
      virtual InnerContext* find_physical_context(unsigned index) = 0;
      virtual size_t get_collective_points(void) const = 0;
    public:
      inline void activate(void) { }
      inline void deactivate(bool) { }
    };

    /**
     * \class CollectiveVersioningBase
     */
    class CollectiveVersioningBase {
    public:
      struct RegionVersioning {
        op::map<std::pair<AddressSpaceID, EqSetTracker*>, FieldMask> trackers;
        RtUserEvent ready_event;
      };
      struct PendingVersioning {
        op::map<LogicalRegion, RegionVersioning> region_versioning;
        size_t remaining_arrivals;
      };
      static void pack_collective_versioning(
          Serializer& rez,
          const op::map<LogicalRegion, RegionVersioning>& to_perform);
      static bool unpack_collective_versioning(
          Deserializer& derez,
          op::map<LogicalRegion, RegionVersioning>& to_perform);
    protected:
      mutable LocalLock versioning_lock;
      std::map<unsigned, PendingVersioning> pending_versioning;
    };

    /**
     * \class CollectiveVersioning
     */
    template<typename OP>
    class CollectiveVersioning : public OP,
                                 public CollectiveVersioningBase {
    public:
      template<typename... Args>
      CollectiveVersioning(Args&&... args) : OP(std::forward<Args>(args)...)
      { }
      CollectiveVersioning(const CollectiveVersioning<OP>& rhs) = delete;
    public:
      CollectiveVersioning<OP>& operator=(const CollectiveVersioning<OP>& rhs) =
          delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      RtEvent rendezvous_collective_versioning_analysis(
          unsigned index, LogicalRegion handle, EqSetTracker* tracker,
          AddressSpaceID space, const FieldMask& mask,
          unsigned parent_req_index);
      void rendezvous_collective_versioning_analysis(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, RegionVersioning>& to_perform);
      virtual void finalize_collective_versioning_analysis(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, RegionVersioning>& to_perform);
    };

    /**
     * \class CollectiveViewCreatorBase
     * The base class that has most of the implementations for
     * collective views creation, modulo the parts that hook in
     * to the operation class.
     */
    class CollectiveViewCreatorBase {
    public:  // Data structures for collective view rendezvous
      struct RendezvousKey {
      public:
        RendezvousKey(void) : region_index(0), analysis(0) { }
        RendezvousKey(unsigned index, unsigned ana)
          : region_index(index), analysis(ana)
        { }
      public:
        inline bool operator<(const RendezvousKey& rhs) const
        {
          if (region_index < rhs.region_index)
            return true;
          if (region_index > rhs.region_index)
            return false;
          return (analysis < rhs.analysis);
        }
        inline bool operator==(const RendezvousKey& rhs) const
        {
          if (region_index != rhs.region_index)
            return false;
          return (analysis == rhs.analysis);
        }
      public:
        unsigned region_index;
        unsigned analysis;
      };
      struct PendingRendezvousKey : public RendezvousKey {
      public:
        PendingRendezvousKey(void)
          : RendezvousKey(), region(LogicalRegion::NO_REGION)
        { }
        PendingRendezvousKey(unsigned index, unsigned ana, LogicalRegion r)
          : RendezvousKey(index, ana), region(r)
        { }
      public:
        inline bool operator<(const PendingRendezvousKey& rhs) const
        {
          if (region_index < rhs.region_index)
            return true;
          if (region_index > rhs.region_index)
            return false;
          if (analysis < rhs.analysis)
            return true;
          if (analysis > rhs.analysis)
            return false;
          return (region < rhs.region);
        }
        inline bool operator==(const PendingRendezvousKey& rhs) const
        {
          if (region_index != rhs.region_index)
            return false;
          if (analysis != rhs.analysis)
            return false;
          return (region == rhs.region);
        }
      public:
        LogicalRegion region;
      };
      struct CollectiveResult : public Collectable {
      public:
        CollectiveResult(
            const std::vector<DistributedID>& dids,
            DistributedID collective_did, RtEvent ready);
        CollectiveResult(
            std::vector<DistributedID>&& dids, DistributedID collective_did,
            RtEvent ready);
        // No-collective instance result
        CollectiveResult(DistributedID instance_did);
        // Temporary result pending response message
        CollectiveResult(const std::vector<DistributedID>& dids);
      public:
        bool matches(const std::vector<DistributedID>& dids) const;
      public:
        const std::vector<DistributedID> individual_dids;
        // Not const so they can be updated by response messages
        DistributedID collective_did;
        RtEvent ready_event;
      };
      struct RendezvousResult : public Collectable {
      public:
        RendezvousResult(
            CollectiveViewCreatorBase* owner, const PendingRendezvousKey& key,
            const InstanceSet& insts, InnerContext* physical_ctx);
        ~RendezvousResult(void);
      public:
        bool matches(const InstanceSet& insts) const;
        static op::vector<std::pair<DistributedID, FieldMask> > init_instances(
            const InstanceSet& insts);
        bool finalize_rendezvous(
            CollectiveMapping* mapping,
            const FieldMapView<CollectiveResult>& views,
            const std::map<DistributedID, size_t>& counts, bool first,
            size_t local);
      public:
        CollectiveViewCreatorBase* const owner;
        InnerContext* const physical_ctx;
        const PendingRendezvousKey key;
        // These are the instances represented for this particular result
        const op::vector<std::pair<DistributedID, FieldMask> > instances;
        const RtUserEvent ready;
      public:
        // These are the places to put the results when ready
        std::vector<CollectiveMapping**> target_mappings;
        std::vector<bool*> target_first_locals;
        std::vector<op::vector<op::FieldMaskMap<InstanceView> >*> target_views;
        std::vector<std::map<InstanceView*, size_t>*> target_arrivals;
      };
      struct CollectiveRendezvous {
      public:
        std::vector<std::pair<AddressSpaceID, RendezvousResult*> > results;
        op::map<DistributedID, FieldMask> groups;
        std::map<DistributedID, size_t> counts;
      };
      struct PendingCollective {
      public:
        PendingCollective(size_t arrivals) : remaining_arrivals(arrivals) { }
      public:
        // Note you can't count the rendezvous results because you can
        // get duplicate arrivals from multiple operations
        std::map<LogicalRegion, CollectiveRendezvous> rendezvous;
        size_t remaining_arrivals;
      };
    public:
      RendezvousResult* find_or_create_rendezvous(
          unsigned index, unsigned analysis, LogicalRegion region,
          const InstanceSet& targets, InnerContext* physical_ctx,
          CollectiveMapping*& analysis_mapping, bool& first_local,
          op::vector<op::FieldMaskMap<InstanceView> >& target_views,
          std::map<InstanceView*, size_t>& collective_arrivals);
      bool remove_pending_rendezvous(RendezvousResult* result);
      static void finalize_collective_mapping(
          CollectiveMapping* mapping, AddressSpaceID owner_space,
          // Can assume that the results are sorted
          std::vector<std::pair<AddressSpaceID, RendezvousResult*> >& results,
          // Instance DID to counts of users
          const std::map<DistributedID, size_t>& counts,
          // The collective views that describes the results for this region
          const FieldMapView<CollectiveResult>& views);
      static void handle_finalize_collective_mapping(Deserializer& derez);
      static void update_groups_and_counts(
          CollectiveRendezvous& target, DistributedID did,
          const FieldMask& mask, size_t count = 1);
      static void pack_collective_rendezvous(
          Serializer& rez,
          const std::map<LogicalRegion, CollectiveRendezvous>& rendezvous);
      static void unpack_collective_rendezvous(
          Deserializer& derez,
          std::map<LogicalRegion, CollectiveRendezvous>& rendezvous);
    protected:
      // Collective instance rendezvous data structures
      mutable LocalLock collective_lock;
      std::map<PendingRendezvousKey, std::vector<RendezvousResult*> >
          pending_rendezvous;
      std::map<RendezvousKey, PendingCollective> pending_collectives;
    };

    /**
     * \class CollectiveViewCreator
     * This class provides common functionality for all index space
     * operations that are going to need to perform rendezvous between
     * point ops/tasks that need to create collective views
     */
    template<typename OP>
    class CollectiveViewCreator : public CollectiveVersioning<OP>,
                                  public CollectiveViewCreatorBase {
    public:
      template<typename... Args>
      CollectiveViewCreator(Args&&... args)
        : CollectiveVersioning<OP>(std::forward<Args>(args)...)
      { }
      CollectiveViewCreator(const CollectiveViewCreator<OP>& rhs) = delete;
    public:
      CollectiveViewCreator<OP>& operator=(
          const CollectiveViewCreator<OP>& rhs) = delete;
    public:
      virtual void activate(void);
      virtual void deactivate(bool free = true);
    public:
      virtual RtEvent convert_collective_views(
          unsigned requirement_index, unsigned analysis_index,
          LogicalRegion region, const InstanceSet& targets,
          InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
          bool& first_local,
          op::vector<op::FieldMaskMap<InstanceView> >& target_views,
          std::map<InstanceView*, size_t>& collective_arrivals);
      // This always needs to happen on the origin node for the operation
      // so we override it in the case of slice task to handle the remote case
      virtual void rendezvous_collective_mapping(
          unsigned requirement_index, unsigned analysis_index,
          LogicalRegion region, RendezvousResult* result, AddressSpaceID source,
          const op::vector<std::pair<DistributedID, FieldMask> >& insts);
      void rendezvous_collective_mapping(
          const RendezvousKey& key,
          std::map<LogicalRegion, CollectiveRendezvous>& rendezvous);
      // In the case of control replication we need to perform additional
      // rendezvous steps across the shards so we override for those cases
      virtual void construct_collective_mapping(
          const RendezvousKey& key,
          std::map<LogicalRegion, CollectiveRendezvous>& rendezvous);
    };

    /**
     * \class CollectiveVersioningRendezvous
     * A gather collective to perform the rendezvous for performing
     * versioning analysis for collective region requirements
     */
    class CollectiveVersioningRendezvous : public GatherCollective {
    public:
      typedef CollectiveVersioningBase::RegionVersioning RegionVersioning;
      class Finalizer {
      public:
        virtual void finalize_collective_versioning(
            unsigned index, unsigned parent_req_index,
            op::map<LogicalRegion, RegionVersioning>& pending_versions) = 0;
      };
    public:
      CollectiveVersioningRendezvous(
          CollectiveID, ReplicateContext* ctx, Operation* op,
          Finalizer* finalizer, ShardID owner, unsigned index);
      CollectiveVersioningRendezvous(
          const CollectiveVersioningRendezvous& rhs) = delete;
      virtual ~CollectiveVersioningRendezvous(void);
    public:
      CollectiveVersioningRendezvous& operator=(
          const CollectiveVersioningRendezvous& rhs) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_VERSIONING_RENDEZVOUS;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
      virtual RtEvent post_gather(void) override;
    public:
      void perform_rendezvous(
          unsigned parent_req_index,
          op::map<LogicalRegion, RegionVersioning>& pending_versions);
    public:
      Operation* const op;
      Finalizer* const finalizer;
      const unsigned index;
    protected:
      op::map<LogicalRegion, RegionVersioning> pending_versions;
      unsigned parent_req_index;
    };

    /**
     * \class ReplCollectiveVersioning
     */
    template<typename OP>
    class ReplCollectiveVersioning
      : public OP,
        public CollectiveVersioningRendezvous::Finalizer {
    public:
      typedef typename OP::RegionVersioning RegionVersioning;
    public:
      ReplCollectiveVersioning(void);
      ReplCollectiveVersioning(const ReplCollectiveVersioning<OP>& rhs) =
          delete;
    public:
      ReplCollectiveVersioning<OP>& operator=(
          const ReplCollectiveVersioning<OP>& rhs) = delete;
    public:
      virtual void deactivate(bool free = true) override;
      virtual void finalize_collective_versioning_analysis(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, RegionVersioning>& to_perform) override;
      virtual void finalize_collective_versioning(
          unsigned index, unsigned parent_req_index,
          op::map<LogicalRegion, RegionVersioning>& pending_versions) override;
    public:
      void create_collective_rendezvous(unsigned requirement_index);
      virtual void shard_off_collective_rendezvous(
          std::set<RtEvent>& done_events);
      virtual void elide_collective_rendezvous(void);
    protected:
      std::map<unsigned, CollectiveVersioningRendezvous*>
          collective_versioning_rendezvous;
    };

    /**
     * \class CollectiveViewRendezvous
     * A gather collective for performing the rendezvous for the creation
     * of collective views across all the shards
     */
    class CollectiveViewRendezvous : public GatherCollective {
    public:
      typedef CollectiveViewCreatorBase::RendezvousKey RendezvousKey;
      typedef CollectiveViewCreatorBase::RendezvousResult RendezvousResult;
      typedef CollectiveViewCreatorBase::CollectiveRendezvous
          CollectiveRendezvous;
      class Finalizer {
      public:
        virtual void finalize_collective_mapping(
            const RendezvousKey& key,
            std::map<LogicalRegion, CollectiveRendezvous>& rendezvous) = 0;
      };
    public:
      CollectiveViewRendezvous(
          CollectiveID, ReplicateContext* ctx, Operation* op,
          Finalizer* finalizer, const RendezvousKey& key, RegionTreeID tid);
      CollectiveViewRendezvous(const CollectiveViewRendezvous& rhs) = delete;
      virtual ~CollectiveViewRendezvous(void);
    public:
      CollectiveViewRendezvous& operator=(const CollectiveViewRendezvous& rhs) =
          delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_VIEW_RENDEZVOUS;
      }
      virtual void pack_collective(Serializer& rez) const override;
      virtual void unpack_collective(Deserializer& derez) override;
      virtual RtEvent post_gather(void) override;
    public:
      void perform_rendezvous(
          std::map<LogicalRegion, CollectiveRendezvous>& rendezvous);
    public:
      const RendezvousKey key;
      Operation* const op;
      Finalizer* const finalizer;
    protected:
      std::map<LogicalRegion, CollectiveRendezvous> rendezvous;
    };

    /**
     * \class ReplCollectiveViewCreator
     * This class provides additional functionality for creating collective
     * views in control replication contexts by helping to manage the
     * rendezvous between the shards.
     */
    template<typename OP>
    class ReplCollectiveViewCreator
      : public ReplCollectiveVersioning<OP>,
        public CollectiveViewRendezvous::Finalizer {
    public:
      typedef typename OP::RendezvousKey RendezvousKey;
      typedef typename OP::CollectiveRendezvous CollectiveRendezvous;
    public:
      ReplCollectiveViewCreator(void);
      ReplCollectiveViewCreator(const ReplCollectiveViewCreator<OP>& rhs) =
          delete;
    public:
      ReplCollectiveViewCreator<OP>& operator=(
          const ReplCollectiveViewCreator<OP>& rhs) = delete;
    public:
      virtual void deactivate(bool free = true) override;
      virtual void construct_collective_mapping(
          const RendezvousKey& key,
          std::map<LogicalRegion, CollectiveRendezvous>& rendezvous) override;
      virtual void finalize_collective_mapping(
          const RendezvousKey& key,
          std::map<LogicalRegion, CollectiveRendezvous>& rendezvous) override;
      void create_collective_rendezvous(
          RegionTreeID tid, unsigned requirement_index,
          unsigned analysis_index = 0);
      virtual void shard_off_collective_rendezvous(
          std::set<RtEvent>& done_events) override;
      virtual void elide_collective_rendezvous(void) override;
    protected:
      std::map<RendezvousKey, CollectiveViewRendezvous*>
          collective_view_rendezvous;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_COLLECTIVE_OPERATION_H__
