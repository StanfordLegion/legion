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

#ifndef __LEGION_FUNCTORS_IMPL_H__
#define __LEGION_FUNCTORS_IMPL_H__

#include "legion/api/functors.h"
#include "legion/operations/operation.h"

namespace Legion {
  namespace Internal {

    /**
     * Identity Projection Functor
     * A class that implements the identity projection function
     */
    class IdentityProjectionFunctor : public ProjectionFunctor {
    public:
      IdentityProjectionFunctor(Legion::Runtime* rt);
      virtual ~IdentityProjectionFunctor(void);
    public:
      using ProjectionFunctor::is_complete;
      using ProjectionFunctor::project;
      virtual LogicalRegion project(
          const Mappable* mappable, unsigned index, LogicalRegion upper_bound,
          const DomainPoint& point) override;
      virtual LogicalRegion project(
          const Mappable* mappable, unsigned index,
          LogicalPartition upper_bound, const DomainPoint& point) override;
      virtual LogicalRegion project(
          LogicalRegion upper_bound, const DomainPoint& point,
          const Domain& launch_domain) override;
      virtual LogicalRegion project(
          LogicalPartition upper_bound, const DomainPoint& point,
          const Domain& launch_domain) override;
      virtual void invert(
          LogicalRegion region, LogicalRegion upper_bound,
          const Domain& launch_domain,
          std::vector<DomainPoint>& ordered_points) override;
      virtual void invert(
          LogicalRegion region, LogicalPartition upper_bound,
          const Domain& launch_domain,
          std::vector<DomainPoint>& ordered_points) override;
      virtual bool is_complete(
          LogicalRegion upper_bound, const Domain& launch_domain) override;
      virtual bool is_complete(
          LogicalPartition upper_bound, const Domain& launch_domain) override;
      virtual bool is_functional(void) const override;
      virtual bool is_exclusive(void) const override;
      virtual unsigned get_depth(void) const override;
    };

    /**
     * \class ProjectionPoint
     * An abstract class for passing to projection functions
     * for recording the results of a projection
     */
    class ProjectionPoint {
    public:
      virtual const DomainPoint& get_domain_point(void) const = 0;
      virtual void set_projection_result(
          unsigned idx, LogicalRegion result) = 0;
      virtual void record_intra_space_dependences(
          unsigned idx, const std::vector<DomainPoint>& region_deps) = 0;
      virtual void record_pointwise_dependence(
          uint64_t previous_context_index, const DomainPoint& previous_point,
          ShardID shard_id) = 0;
      virtual const Operation* as_operation(void) const = 0;
    };

    /**
     * \class ProjectionFunction
     * A class for wrapping projection functors
     */
    class ProjectionFunction {
    public:
      ProjectionFunction(ProjectionID pid, ProjectionFunctor* functor);
      ProjectionFunction(const ProjectionFunction& rhs) = delete;
      ~ProjectionFunction(void);
    public:
      ProjectionFunction& operator=(const ProjectionFunction& rhs) = delete;
    public:
      void prepare_for_shutdown(void);
    public:
      // The old path explicitly for tasks
      LogicalRegion project_point(
          Task* task, unsigned idx, const Domain& launch_domain,
          const DomainPoint& point);
      void project_points(
          const RegionRequirement& req, unsigned idx,
          const Domain& launch_domain,
          const std::vector<PointTask*>& point_tasks,
          const std::vector<PointwiseDependence>* pointwise,
          const size_t total_shards, bool replaying);
      // Generalized and annonymized
      void project_points(
          Operation* op, unsigned idx, const RegionRequirement& req,
          const Domain& launch_domain,
          const std::vector<ProjectionPoint*>& points,
          const std::vector<PointwiseDependence>* pointwise,
          const size_t total_shards, bool replaying);
      // Find inversions for pointwise dependence analysis
      void find_inversions(
          OpKind op_kind, UniqueID uid, unsigned region_index,
          const RegionRequirement& req, IndexSpaceNode* domain,
          const std::vector<LogicalRegion>& regions,
          std::map<LogicalRegion, std::vector<DomainPoint> >& dependences);
    protected:
      // Old checking code explicitly for tasks
      void check_projection_region_result(
          LogicalRegion upper_bound, const Task* task, unsigned idx,
          LogicalRegion result) const;
      void check_projection_partition_result(
          LogicalPartition upper_bound, const Task* task, unsigned idx,
          LogicalRegion result) const;
      // Annonymized checking code
      void check_projection_region_result(
          LogicalRegion upper_bound, Operation* op, unsigned idx,
          LogicalRegion result) const;
      void check_projection_partition_result(
          LogicalPartition upper_bound, Operation* op, unsigned idx,
          LogicalRegion result) const;
      // Checking for inversion
      void check_inversion(
          const ProjectionPoint* point, unsigned idx,
          const std::vector<DomainPoint>& ordered_points,
          const Domain& launch_domain, bool allow_empty = false);
      void check_containment(
          const ProjectionPoint* point, unsigned idx,
          const std::vector<DomainPoint>& ordered_points);
      void check_inversion(
          OpKind op_kind, UniqueID uid, unsigned idx,
          const std::vector<DomainPoint>& ordered_points,
          const Domain& launch_domain, bool allow_empty = false);
      void check_containment(
          OpKind op_kind, UniqueID uid, unsigned idx, const DomainPoint& point,
          const std::vector<DomainPoint>& ordered_points);
    public:
      bool is_complete(
          RegionTreeNode* node, Operation* op, unsigned index,
          IndexSpaceNode* projection_space) const;
      ProjectionNode* construct_projection_tree(
          Operation* op, unsigned index, const RegionRequirement& req,
          ShardID local_shard, RegionTreeNode* root,
          const ProjectionInfo& proj_info);
      static void add_to_projection_tree(
          LogicalRegion region, RegionTreeNode* root,
          std::map<RegionTreeNode*, ProjectionNode*>& node_map,
          ShardID owner_shard);
    public:
      const unsigned depth;
      const bool is_exclusive;
      const bool is_functional;
      const bool is_invertible;
      const ProjectionID projection_id;
      ProjectionFunctor* const functor;
    protected:
      mutable LocalLock projection_reservation;
    };

    /**
     * \class CyclicShardingFunctor
     * The cyclic sharding functor just round-robins the points
     * onto the available set of shards
     */
    class CyclicShardingFunctor : public ShardingFunctor {
    public:
      CyclicShardingFunctor(void);
      CyclicShardingFunctor(const CyclicShardingFunctor& rhs) = delete;
      virtual ~CyclicShardingFunctor(void);
    public:
      CyclicShardingFunctor& operator=(const CyclicShardingFunctor& rhs) =
          delete;
    public:
      template<int DIM>
      size_t linearize_point(
          const Realm::IndexSpace<DIM, coord_t>& is,
          const Realm::Point<DIM, coord_t>& point) const;
    public:
      virtual ShardID shard(
          const DomainPoint& point, const Domain& full_space,
          const size_t total_shards) override;
    };

    /**
     * \class ShardingFunction
     * The sharding function class wraps a sharding functor and will
     * cache results for queries so that we don't need to constantly
     * be inverting the results of the sharding functor.
     */
    class ShardingFunction {
    public:
      struct ShardKey {
      public:
        ShardKey(void)
          : sid(0), full_space(IndexSpace::NO_SPACE),
            shard_space(IndexSpace::NO_SPACE)
        { }
        ShardKey(ShardID s, IndexSpace f, IndexSpace sh)
          : sid(s), full_space(f), shard_space(sh)
        { }
      public:
        inline bool operator<(const ShardKey& rhs) const
        {
          if (sid < rhs.sid)
            return true;
          if (sid > rhs.sid)
            return false;
          if (full_space < rhs.full_space)
            return true;
          if (full_space > rhs.full_space)
            return false;
          return shard_space < rhs.shard_space;
        }
        inline bool operator==(const ShardKey& rhs) const
        {
          if (sid != rhs.sid)
            return false;
          if (full_space != rhs.full_space)
            return false;
          return shard_space == rhs.shard_space;
        }
      public:
        ShardID sid;
        IndexSpace full_space, shard_space;
      };
    public:
      ShardingFunction(
          ShardingFunctor* functor, ShardManager* manager,
          ShardingID sharding_id, bool skip_checks = false,
          bool own_functor = false);
      ShardingFunction(const ShardingFunction& rhs) = delete;
      virtual ~ShardingFunction(void);
    public:
      ShardingFunction& operator=(const ShardingFunction& rhs) = delete;
    public:
      ShardID find_owner(
          const DomainPoint& point, const Domain& sharding_space);
      IndexSpace find_shard_space(
          ShardID shard, IndexSpaceNode* full_space, IndexSpace sharding_space,
          Provenance* provenance);
      bool find_shard_participants(
          IndexSpaceNode* full_space, IndexSpace sharding_space,
          std::vector<ShardID>& participants);
      bool has_participants(
          ShardID shard, IndexSpaceNode* full_space, IndexSpace sharding_space);
    public:
      ShardingFunctor* const functor;
      ShardManager* const manager;
      const ShardingID sharding_id;
      const bool use_points;
      const bool skip_checks;
      const bool own_functor;
    protected:
      mutable LocalLock sharding_lock;
      std::map<ShardKey, IndexSpace /*result*/> shard_index_spaces;
      std::map<std::pair<IndexSpace, IndexSpace>, std::vector<ShardID> >
          shard_participants;
    };

    /**
     * \class ZeroColoringFunctor
     * The zero coloring functor maps all points to color zero
     * so that they all have to be concurrent with each other
     */
    class ZeroColoringFunctor : public ConcurrentColoringFunctor {
    public:
      ZeroColoringFunctor(void) { }
      virtual ~ZeroColoringFunctor(void) { }
    public:
      virtual Color color(
          const DomainPoint& point, const Domain& index_domain) override
      {
        return 0;
      }
      virtual bool supports_max_color(void) override { return true; }
      virtual Color max_color(const Domain& index_domain) override { return 0; }
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_FUNCTORS_IMPL_H__
