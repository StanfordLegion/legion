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

#ifndef __LEGION_PROJECTION_H__
#define __LEGION_PROJECTION_H__

#include "legion/kernel/garbage_collection.h"
#include "legion/api/requirements.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /**
     * \class ProjectionInfo
     * Projection information for index space requirements
     */
    class ProjectionInfo {
    public:
      ProjectionInfo(
          const RegionRequirement* req, IndexSpaceNode* launch_space,
          ShardingFunction* func, IndexSpace shard_space);
    public:
      inline bool is_projecting(void) const { return (projection != nullptr); }
      inline bool is_sharding(void) const
      {
        return (sharding_function != nullptr);
      }
      bool is_complete_projection(
          RegionTreeNode* node, const LogicalUser& user) const;
    public:
      ProjectionFunction* projection;
      ProjectionType projection_type;
      IndexSpaceNode* projection_space;
      ShardingFunction* sharding_function;
      IndexSpaceNode* sharding_space;
    };

    /**
     * \class ShardedColorMap
     * This data structure is a look-up table for mapping colors in a
     * disjoint and complete partition to the nearest shard that knows
     * about them. It is the only data structure that we create anywhere
     * which might be on the order of the number of nodes/shards and
     * stored on every node so we deduplicate it across all its uses.
     */
    class ShardedColorMap : public Collectable {
    public:
      ShardedColorMap(std::unordered_map<LegionColor, ShardID>&& map)
        : color_shards(map)
      { }
    public:
      inline bool empty(void) const { return color_shards.empty(); }
      inline size_t size(void) const { return color_shards.size(); }
      ShardID at(LegionColor color) const;
      void pack(Serializer& rez);
      static void pack_empty(Serializer& rez);
      static ShardedColorMap* unpack(Deserializer& derez);
    public:
      // Must remain constant since many things can refer to this
      const std::unordered_map<LegionColor, ShardID> color_shards;
    };

    /**
     * \class ProjectionNode
     * A projection node represents a summary of the regions and partitions
     * accessed by a projection function from a particular node in the
     * region tree. In the case of control replication, it specifically
     * stores the accesses performed by the local shard, and stores at
     * least one nullptr child for any aliasing children from different
     * shards to facilitate testing for any close operations that might
     * be required between shards. We can also test a projection node
     * to see if it can be converted to a refinement node (e.g. that
     * is has all disjoint-complete partitions).
     */
    class ProjectionNode : public Collectable {
    public:
      /**
       * This class defines an interval tree on colors that we can
       * that we can use to test if a projection node interferes with
       * children from a remote shard. The tree is defined for the ranges
       * of colors represented by remote shards. These ranges are
       * exclusive from our local children. The reason to use
       * ranges here is to compress the representation of the colors
       * from all the remote shards. We need a way to test if children
       * are represented on remote shards, but we don't care what's
       * in their subtrees. The ranges are semi-inclusive [start,end)
       * Note that the ranges by definition cannot overlap with eachother
       * It's important to realized that the only reason that this
       * compression works is that we linearize colors in N-d color
       * spaces using Morton codes which gives good locality for
       * nearest neighbors and encourages this compressibility so
       * we can efficiently store the children as ranges to query.
       */
      class IntervalTree {
      public:
        inline void swap(IntervalTree& rhs) { ranges.swap(rhs.ranges); }
        inline bool empty(void) const { return ranges.empty(); }
        void add_child(LegionColor color);
        void remove_child(LegionColor color);
        void add_range(LegionColor start, LegionColor stop);
        bool has_child(LegionColor color) const;
        void serialize(Serializer& rez) const;
        void deserialize(Deserializer& derez);
      public:
        std::map<LegionColor /*start*/, LegionColor /*end*/> ranges;
      };
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
      /**
       * This class defines a compact way of representing a set of shards.
       * It maintains two different representations of the set depending
       * on how many entries it contains. It stores the names of shards
       * sorted in order in a contiguous vector of entries up to the point
       * that the size of the space needed to store the entries exceeds the
       * size of the bitmask required to represent to encode the entries.
       * Above this size the set is encoded as a bitmask.
       */
      class ShardSet {
      public:
        ShardSet(void);
        ShardSet(const ShardSet& rhs) = delete;
        ~ShardSet(void);
      public:
        ShardSet& operator=(const ShardSet& rhs) = delete;
      public:
        void insert(ShardID shard, unsigned total_shards);
        ShardID find_nearest_shard(
            ShardID local_shard, unsigned total_shards) const;
      private:
        ShardID find_nearest(
            ShardID local_shard, unsigned total_shards, const ShardID* buffer,
            unsigned buffer_size) const;
        static unsigned find_distance(ShardID one, ShardID two, unsigned total);
      public:
        void serialize(Serializer& rez, unsigned total_shards) const;
        void deserialize(Deserializer& derez, unsigned total_shards);
      private:
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsizeof-pointer-div"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsizeof-pointer-div"
#endif
        // Stupid compilers, I mean what I say, this is not a fucking error
        // I want this to be exactly the same size as ShardID*
        static constexpr unsigned MAX_VALUES =
            sizeof(ShardID*) / sizeof(ShardID);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
        static_assert(MAX_VALUES > 0, "very strange machine");
        union {
          ShardID* buffer;
          ShardID values[MAX_VALUES];
        } set;
        // number of entries in the buffer
        unsigned size;
        // total possible entries in the buffer
        unsigned max;
      };
#endif  // LEGION_NAME_BASED_CHILDREN_SHARDS
      // These structures are used for exchanging summary information
      // between different shards with control replication
      struct RegionSummary {
        ProjectionNode::IntervalTree children;
        std::vector<ShardID> users;
      };
      struct PartitionSummary {
        ProjectionNode::IntervalTree children;
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
        // If we're disjoint and complete we also track the sets
        // of shards that know about each of the children as well
        // so we can record the one nearest for each shard
        std::unordered_map<LegionColor, ShardSet>
            disjoint_complete_child_shards;
#endif  // LEGION_NAME_BASED_CHILDREN_SHARDS
      };
    public:
      virtual ~ProjectionNode(void) { };
      virtual ProjectionRegion* as_region_projection(void) { return nullptr; }
      virtual ProjectionPartition* as_partition_projection(void)
      {
        return nullptr;
      }
      virtual bool is_disjoint(void) const = 0;
      virtual bool is_leaves_only(void) const = 0;
      virtual bool is_unique_shards(void) const = 0;
      virtual bool interferes(
          ProjectionNode* other, ShardID local, bool& dominates) const = 0;
      virtual bool pointwise_dominates(const ProjectionNode* other) const = 0;
      virtual void extract_shard_summaries(
          bool supports_name_based_analysis, ShardID local_shard,
          size_t total_shards, std::map<LogicalRegion, RegionSummary>& regions,
          std::map<LogicalPartition, PartitionSummary>& partitions) const = 0;
      virtual void update_shard_summaries(
          bool supports_name_based_analysis, ShardID local_shard,
          size_t total_shards, std::map<LogicalRegion, RegionSummary>& regions,
          std::map<LogicalPartition, PartitionSummary>& partitions) = 0;
    public:
      IntervalTree shard_children;
    };

    class ProjectionRegion : public ProjectionNode {
    public:
      ProjectionRegion(RegionNode* node);
      ProjectionRegion(const ProjectionRegion& rhs) = delete;
      virtual ~ProjectionRegion(void);
    public:
      ProjectionRegion& operator=(const ProjectionRegion& rhs) = delete;
    public:
      virtual ProjectionRegion* as_region_projection(void) override
      {
        return this;
      }
      virtual bool is_disjoint(void) const override;
      virtual bool is_leaves_only(void) const override;
      virtual bool is_unique_shards(void) const override;
      virtual bool interferes(
          ProjectionNode* other, ShardID local, bool& dominates) const override;
      virtual bool pointwise_dominates(
          const ProjectionNode* other) const override;
      virtual void extract_shard_summaries(
          bool supports_name_based_analysis, ShardID local_shard,
          size_t total_shards, std::map<LogicalRegion, RegionSummary>& regions,
          std::map<LogicalPartition, PartitionSummary>& partitions)
          const override;
      virtual void update_shard_summaries(
          bool supports_name_based_analysis, ShardID local_shard,
          size_t total_shards, std::map<LogicalRegion, RegionSummary>& regions,
          std::map<LogicalPartition, PartitionSummary>& partitions) override;
      bool has_interference(
          ProjectionRegion* other, ShardID local, bool& dominates) const;
      bool has_pointwise_dominance(const ProjectionRegion* other) const;
      void add_user(ShardID shard);
      void add_child(ProjectionPartition* child);
    public:
      RegionNode* const region;
      std::unordered_map<LegionColor, ProjectionPartition*> local_children;
      std::vector<ShardID> shard_users;  // this vector is sorted
    };

    class ProjectionPartition : public ProjectionNode {
    public:
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
      ProjectionPartition(PartitionNode* node, ShardedColorMap* map = nullptr);
#else
      ProjectionPartition(PartitionNode* node);
#endif
      ProjectionPartition(const ProjectionPartition& rhs) = delete;
      virtual ~ProjectionPartition(void);
    public:
      ProjectionPartition& operator=(const ProjectionPartition& rhs) = delete;
    public:
      virtual ProjectionPartition* as_partition_projection(void) override
      {
        return this;
      }
      virtual bool is_disjoint(void) const override;
      virtual bool is_leaves_only(void) const override;
      virtual bool is_unique_shards(void) const override;
      virtual bool interferes(
          ProjectionNode* other, ShardID local, bool& dominates) const override;
      virtual bool pointwise_dominates(
          const ProjectionNode* other) const override;
      virtual void extract_shard_summaries(
          bool supports_name_based_analysis, ShardID local_shard,
          size_t total_shards, std::map<LogicalRegion, RegionSummary>& regions,
          std::map<LogicalPartition, PartitionSummary>& partitions)
          const override;
      virtual void update_shard_summaries(
          bool supports_name_based_analysis, ShardID local_shard,
          size_t total_shards, std::map<LogicalRegion, RegionSummary>& regions,
          std::map<LogicalPartition, PartitionSummary>& partitions) override;
      bool has_interference(
          ProjectionPartition* other, ShardID local, bool& dominates) const;
      bool has_pointwise_dominance(const ProjectionPartition* other) const;
      void add_child(ProjectionRegion* child);
    public:
      PartitionNode* const partition;
      std::unordered_map<LegionColor, ProjectionRegion*> local_children;
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
      // This is only filled in if we support name-based dependence
      // analysis (disjoint and all users at the leaves)
      ShardedColorMap* name_based_children_shards;
#endif
    };

    /**
     * \class ProjectionSummary
     * A projection summary tracks the meta-data associated with a
     * particular projection including the projection tree that was
     * produced to represent it.
     */
    class ProjectionSummary : public Collectable {
    public:
      // Non-replicated
      ProjectionSummary(
          const ProjectionInfo& info, ProjectionNode* node, Operation* op,
          unsigned index, const RegionRequirement& req, LogicalState* owner);
      // Replicated for projection functor 0
      ProjectionSummary(
          const ProjectionInfo& info, ProjectionNode* node, Operation* op,
          unsigned index, const RegionRequirement& req, LogicalState* owner,
          bool disjoint, bool unique);
      // General replicated
      ProjectionSummary(
          const ProjectionInfo& info, ProjectionNode* node, Operation* op,
          unsigned index, const RegionRequirement& req, LogicalState* owner,
          ReplicateContext* context);
      ProjectionSummary(const ProjectionSummary& rhs) = delete;
      ~ProjectionSummary(void);
    public:
      ProjectionSummary& operator=(const ProjectionSummary& rhs) = delete;
    public:
      bool matches(
          const ProjectionInfo& rhs, const RegionRequirement& req) const;
      inline bool is_complete(void) const { return complete; }
      bool is_disjoint(void);
      bool can_perform_name_based_self_analysis(void);
      bool has_unique_shard_users(void);
      ProjectionNode* get_tree(void);
    public:
      LogicalState* const owner;
      IndexSpaceNode* const domain;
      ProjectionFunction* const projection;
      ShardingFunction* const sharding;
      IndexSpaceNode* const sharding_domain;
      const size_t arglen;
      void* const args;
    private:
      // These members are not actually ready until the exchange has
      // completed which is why they are private to ensure everything
      // goes through the getter interfaces which will check that the
      // exchange has complete before allowing access
      ProjectionNode* const tree;
      // For control replication contexts we might have an outstanding
      // exchange that is being used to finalize the tree and update
      // the properties of the tree
      ProjectionTreeExchange* exchange;
      // We track a few different properties of this index space launch
      // that are useful for various different analyses and kinds of
      // comparisons between index space launches
      // Whether we know all the points are disjoint from each other
      // based privileges of the projection and the projection function
      bool disjoint;
      // Whether this projection tree is complete or not according to
      // the projection functor
      const bool complete;
      // Whether this projection summary can be analyzed against itself
      // using name-based dependence analysis which is that same as
      // having sub-regions described using a disjoint-only subtree
      // and all accesses at the leaves of the tree
      // Note that individual points in the same launch can still use
      // the same sub-regions here
      bool permits_name_based_self_analysis;
      // Whether each region has a unique set of shards users
      bool unique_shard_users;
    };

    /**
     * \class ProjectionTreeExchange
     * This class provides a way of exchanging the projection trees
     * data structures between the shards in a way that is memory
     * efficient and won't involve materializing all the data for
     * each subtree on every node.
     */
    class ProjectionTreeExchange : public AllGatherCollective<false> {
    public:
      ProjectionTreeExchange(
          ProjectionNode* n, ReplicateContext* ctx, CollectiveIndexLocation loc,
          bool& disjoint, bool& permits_name_based_self_analysis,
          bool& unique_shards);
      ProjectionTreeExchange(const ProjectionTreeExchange& rhs) = delete;
      ~ProjectionTreeExchange(void);
    public:
      ProjectionTreeExchange& operator=(const ProjectionTreeExchange&) = delete;
    public:
      virtual MessageKind get_message_kind(void) const override
      {
        return SEND_CONTROL_REPLICATION_PROJECTION_TREE_EXCHANGE;
      }
      virtual void pack_collective_stage(
          ShardID target, Serializer& rez, int stage) override;
      virtual void unpack_collective_stage(
          Deserializer& derez, int stage) override;
      virtual RtEvent post_complete_exchange(void) override;
    public:
      ProjectionNode* const node;
      bool& disjoint;
      bool& permits_name_based_self_analysis;
      bool& unique_shards;
      bool leaves_only;
    protected:
#ifdef LEGION_NAME_BASED_CHILDREN_SHARDS
      typedef ProjectionNode::ShardSet ShardSet;
#endif
      typedef ProjectionNode::RegionSummary RegionSummary;
      typedef ProjectionNode::PartitionSummary PartitionSummary;
      std::map<LogicalRegion, RegionSummary> region_summaries;
      std::map<LogicalPartition, PartitionSummary> partition_summaries;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PROJECTION_H__
