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

#include "legion/analysis/projection.h"
#include "legion/api/data.h"
#include "legion/api/exception.h"
#include "legion/api/functors_impl.h"
#include "legion/api/mapping.h"
#include "legion/managers/shard.h"
#include "legion/nodes/index.h"
#include "legion/nodes/region.h"
#include "legion/operations/pointwise.h"
#include "legion/tasks/point.h"
#include "legion/utilities/privileges.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // ProjectionFunctor
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  ProjectionFunctor::ProjectionFunctor(void) : runtime(nullptr)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  ProjectionFunctor::ProjectionFunctor(Runtime* rt) : runtime(rt)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  ProjectionFunctor::~ProjectionFunctor(void)
  //--------------------------------------------------------------------------
  { }

  // FIXME: This exists for backwards compatibility but it is tripping
  // over our own deprecation warnings. Turn those off inside this method.
  LEGION_DISABLE_DEPRECATED_WARNINGS

  //--------------------------------------------------------------------------
  LogicalRegion ProjectionFunctor::project(
      const Mappable* mappable, unsigned index, LogicalRegion upper_bound,
      const DomainPoint& point)
  //--------------------------------------------------------------------------
  {
    if (is_functional())
    {
      switch (mappable->get_mappable_type())
      {
        case LEGION_TASK_MAPPABLE:
          {
            const Task* task = mappable->as_task();
            return project(upper_bound, point, task->index_domain);
          }
        case LEGION_COPY_MAPPABLE:
          {
            const Copy* copy = mappable->as_copy();
            return project(upper_bound, point, copy->index_domain);
          }
        case LEGION_INLINE_MAPPABLE:
        case LEGION_ACQUIRE_MAPPABLE:
        case LEGION_RELEASE_MAPPABLE:
        case LEGION_CLOSE_MAPPABLE:
          {
            const Domain launch_domain(point, point);
            return project(upper_bound, point, launch_domain);
          }
        case LEGION_FILL_MAPPABLE:
          {
            const Fill* fill = mappable->as_fill();
            return project(upper_bound, point, fill->index_domain);
          }
        case LEGION_PARTITION_MAPPABLE:
          {
            const Partition* part = mappable->as_partition();
            return project(upper_bound, point, part->index_domain);
          }
        case LEGION_MUST_EPOCH_MAPPABLE:
          {
            const MustEpoch* must = mappable->as_must_epoch();
            return project(upper_bound, point, must->launch_domain);
          }
        default:
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Unknown mappable type passed to projection "
                  << "functor! You must override the default "
                  << "implementations of the non-deprecated 'project' methods!";
            error.raise();
          }
      }
    }
    else
    {
      Warning warning;
      warning << "There are new methods for projection functors that must be "
              << "overriden. Calling deprecated methods for now!";
      warning.raise();
      switch (mappable->get_mappable_type())
      {
        case LEGION_TASK_MAPPABLE:
          return project(
              0 /*dummy ctx*/, const_cast<Task*>(mappable->as_task()), index,
              upper_bound, point);
        default:
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Unknown mappable type passed to projection "
                  << "functor! You must override the default "
                  << "implementations of the non-deprecated 'project' methods!";
            error.raise();
          }
      }
    }
    return LogicalRegion::NO_REGION;
  }

  //--------------------------------------------------------------------------
  LogicalRegion ProjectionFunctor::project(
      const Mappable* mappable, unsigned index, LogicalPartition upper_bound,
      const DomainPoint& point)
  //--------------------------------------------------------------------------
  {
    if (is_functional())
    {
      switch (mappable->get_mappable_type())
      {
        case LEGION_TASK_MAPPABLE:
          {
            const Task* task = mappable->as_task();
            return project(upper_bound, point, task->index_domain);
          }
        case LEGION_COPY_MAPPABLE:
          {
            const Copy* copy = mappable->as_copy();
            return project(upper_bound, point, copy->index_domain);
          }
        case LEGION_INLINE_MAPPABLE:
        case LEGION_ACQUIRE_MAPPABLE:
        case LEGION_RELEASE_MAPPABLE:
        case LEGION_CLOSE_MAPPABLE:
          {
            const Domain launch_domain(point, point);
            return project(upper_bound, point, launch_domain);
          }
        case LEGION_FILL_MAPPABLE:
          {
            const Fill* fill = mappable->as_fill();
            return project(upper_bound, point, fill->index_domain);
          }
        case LEGION_PARTITION_MAPPABLE:
          {
            const Partition* part = mappable->as_partition();
            return project(upper_bound, point, part->index_domain);
          }
        case LEGION_MUST_EPOCH_MAPPABLE:
          {
            const MustEpoch* must = mappable->as_must_epoch();
            return project(upper_bound, point, must->launch_domain);
          }
        default:
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Unknown mappable type passed to projection "
                  << "functor! You must override the default "
                  << "implementations of the non-deprecated 'project' methods!";
            error.raise();
          }
      }
    }
    else
    {
      Warning warning;
      warning << "There are new methods for projection functors that must be "
              << "overriden. Calling deprecated methods for now!";
      warning.raise();
      switch (mappable->get_mappable_type())
      {
        case LEGION_TASK_MAPPABLE:
          return project(
              0 /*dummy ctx*/, const_cast<Task*>(mappable->as_task()), index,
              upper_bound, point);
        default:
          {
            Error error(LEGION_INTERFACE_EXCEPTION);
            error << "Unknown mappable type passed to projection "
                  << "functor! You must override the default "
                  << "implementations of the non-deprecated 'project' methods!";
            error.raise();
          }
      }
    }
    return LogicalRegion::NO_REGION;
  }

  LEGION_REENABLE_DEPRECATED_WARNINGS

  //--------------------------------------------------------------------------
  LogicalRegion ProjectionFunctor::project(
      LogicalRegion upper_bound, const DomainPoint& point,
      const Domain& launch_domain)
  //--------------------------------------------------------------------------
  {
    // Must be override by derived classes
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Missing override of ProjectionFunctor::project";
    error.raise();
  }

  //--------------------------------------------------------------------------
  LogicalRegion ProjectionFunctor::project(
      LogicalPartition upper_bound, const DomainPoint& point,
      const Domain& launch_domain)
  //--------------------------------------------------------------------------
  {
    // Must be override by derived classes
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Missing override of ProjectionFunctor::project";
    error.raise();
  }

  //--------------------------------------------------------------------------
  LogicalRegion ProjectionFunctor::project(
      LogicalRegion upper_bound, const DomainPoint& point,
      const Domain& launch_domain, const void* args, size_t size)
  //--------------------------------------------------------------------------
  {
    return project(upper_bound, point, launch_domain);
  }

  //--------------------------------------------------------------------------
  LogicalRegion ProjectionFunctor::project(
      LogicalPartition upper_bound, const DomainPoint& point,
      const Domain& launch_domain, const void* args, size_t size)
  //--------------------------------------------------------------------------
  {
    return project(upper_bound, point, launch_domain);
  }

  //--------------------------------------------------------------------------
  LogicalRegion ProjectionFunctor::project(
      Context ctx, Task* task, unsigned index, LogicalRegion upper_bound,
      const DomainPoint& point)
  //--------------------------------------------------------------------------
  {
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Invocation of deprecated projection functor without an "
          << "override";
    error.raise();
  }

  //--------------------------------------------------------------------------
  LogicalRegion ProjectionFunctor::project(
      Context ctx, Task* task, unsigned index, LogicalPartition upper_bound,
      const DomainPoint& point)
  //--------------------------------------------------------------------------
  {
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Invocation of deprecated projection functor without an "
          << "override";
    error.raise();
  }

  //--------------------------------------------------------------------------
  void ProjectionFunctor::invert(
      LogicalRegion region, LogicalRegion upper, const Domain& launch_domain,
      std::vector<DomainPoint>& ordered_points)
  //--------------------------------------------------------------------------
  {
    // Must be override by derived classes
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Missing override of ProjectionFunctor::invert";
    error.raise();
  }

  //--------------------------------------------------------------------------
  void ProjectionFunctor::invert(
      LogicalRegion region, LogicalPartition upper, const Domain& launch_domain,
      std::vector<DomainPoint>& ordered_points)
  //--------------------------------------------------------------------------
  {
    // Must be override by derived classes
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Missing override of ProjectionFunctor::invert";
    error.raise();
  }

  //--------------------------------------------------------------------------
  bool ProjectionFunctor::is_complete(
      LogicalRegion upper_bound, const Domain& launch_domain)
  //--------------------------------------------------------------------------
  {
    return false;
  }

  //--------------------------------------------------------------------------
  bool ProjectionFunctor::is_complete(
      LogicalPartition upper_bound, const Domain& launch_domain)
  //--------------------------------------------------------------------------
  {
    return false;
  }

  //--------------------------------------------------------------------------
  bool ProjectionFunctor::is_complete(
      Mappable* mappable, unsigned index, LogicalRegion upper_bound,
      const Domain& launch_domain)
  //--------------------------------------------------------------------------
  {
    return false;
  }

  //--------------------------------------------------------------------------
  bool ProjectionFunctor::is_complete(
      Mappable* mappable, unsigned index, LogicalPartition upper_bound,
      const Domain& launch_domain)
  //--------------------------------------------------------------------------
  {
    return false;
  }

  /////////////////////////////////////////////////////////////
  // ShardingFunctor
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  ShardingFunctor::ShardingFunctor(void)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  ShardingFunctor::~ShardingFunctor(void)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  ShardID ShardingFunctor::shard(
      const DomainPoint& index_point, const Domain& index_domain,
      const size_t total_shards)
  //--------------------------------------------------------------------------
  {
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Invocation of 'ShardingFunctor::shard' method "
          << "without a user-provided override";
    error.raise();
  }

  //--------------------------------------------------------------------------
  DomainPoint ShardingFunctor::shard_points(
      const DomainPoint& index_point, const Domain& index_domain,
      const std::vector<DomainPoint>& shard_points, const Domain& shard_domain)
  //--------------------------------------------------------------------------
  {
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Invocation of 'ShardingFunctor::shard_points' method "
          << "without a user-provided override";
    error.raise();
  }

  //--------------------------------------------------------------------------
  void ShardingFunctor::invert(
      ShardID shard, const Domain& sharding_domain, const Domain& index_domain,
      const size_t total_shards, std::vector<DomainPoint>& points)
  //--------------------------------------------------------------------------
  {
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Invocation of 'ShardingFunctor::invert' method "
          << "without a user-provided override";
    error.raise();
  }

  //--------------------------------------------------------------------------
  void ShardingFunctor::invert_points(
      const DomainPoint& shard_point,
      const std::vector<DomainPoint>& shard_points, const Domain& shard_domain,
      const Domain& index_domain, const Domain& sharding_domain,
      std::vector<DomainPoint>& index_points)
  //--------------------------------------------------------------------------
  {
    Error error(LEGION_INTERFACE_EXCEPTION);
    error << "Invocation of 'ShardingFunctor::invert_points' method "
          << "without a user-provided override";
    error.raise();
  }

  /////////////////////////////////////////////////////////////
  // Concurrent Coloring Functor
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  ConcurrentColoringFunctor::ConcurrentColoringFunctor(void)
  //--------------------------------------------------------------------------
  { }

  //--------------------------------------------------------------------------
  ConcurrentColoringFunctor::~ConcurrentColoringFunctor(void)
  //--------------------------------------------------------------------------
  { }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Identity Projection Functor
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IdentityProjectionFunctor::IdentityProjectionFunctor(Legion::Runtime* rt)
      : ProjectionFunctor(rt)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IdentityProjectionFunctor::~IdentityProjectionFunctor(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LogicalRegion IdentityProjectionFunctor::project(
        const Mappable* mappable, unsigned index, LogicalRegion upper_bound,
        const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      // We know we don't use the domain so we can fake it
      Domain launch_domain;
      return project(upper_bound, point, launch_domain);
    }

    //--------------------------------------------------------------------------
    LogicalRegion IdentityProjectionFunctor::project(
        const Mappable* mappable, unsigned index, LogicalPartition upper_bound,
        const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      // We know we don't use the domain so we can fake it
      Domain launch_domain;
      return project(upper_bound, point, launch_domain);
    }

    //--------------------------------------------------------------------------
    LogicalRegion IdentityProjectionFunctor::project(
        LogicalRegion upper_bound, const DomainPoint& point,
        const Domain& launch_domain)
    //--------------------------------------------------------------------------
    {
      return upper_bound;
    }

    //--------------------------------------------------------------------------
    LogicalRegion IdentityProjectionFunctor::project(
        LogicalPartition up_bound, const DomainPoint& point,
        const Domain& launch_domain)
    //--------------------------------------------------------------------------
    {
      return runtime->get_logical_subregion_by_color(up_bound, point);
    }

    //--------------------------------------------------------------------------
    void IdentityProjectionFunctor::invert(
        LogicalRegion region, LogicalRegion upper_bound,
        const Domain& launch_domain, std::vector<DomainPoint>& ordered_points)
    //--------------------------------------------------------------------------
    {
      legion_assert(region == upper_bound);
      // This is a special case for the ordered mapping of point tasks in
      // the case where we used to try to premap regions for an index task
      // launch where all the points mapped the same region with read-write
      // Just enumerate the points in order for the domain
      ordered_points.reserve(launch_domain.get_volume());
      for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
        ordered_points.emplace_back(itr.p);
    }

    //--------------------------------------------------------------------------
    void IdentityProjectionFunctor::invert(
        LogicalRegion region, LogicalPartition upper_bound,
        const Domain& launch_domain, std::vector<DomainPoint>& ordered_points)
    //--------------------------------------------------------------------------
    {
      legion_assert(
          runtime->get_parent_logical_partition(region) == upper_bound);
      const DomainPoint point = runtime->get_logical_region_color_point(region);
      // Need to check if it is in the launch domain
      if (launch_domain.contains(point))
        ordered_points.emplace_back(point);
    }

    //--------------------------------------------------------------------------
    bool IdentityProjectionFunctor::is_complete(
        LogicalRegion upper_bound, const Domain& launch_domain)
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    bool IdentityProjectionFunctor::is_complete(
        LogicalPartition upper_bound, const Domain& launch_domain)
    //--------------------------------------------------------------------------
    {
      const Domain color_space_domain =
          runtime->get_index_partition_color_space(
              upper_bound.get_index_partition());
      return (
          (color_space_domain == launch_domain) ||
          (color_space_domain.get_volume() == launch_domain.get_volume()));
    }

    //--------------------------------------------------------------------------
    bool IdentityProjectionFunctor::is_functional(void) const
    //--------------------------------------------------------------------------
    {
      return true;
    }

    //--------------------------------------------------------------------------
    bool IdentityProjectionFunctor::is_exclusive(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    unsigned IdentityProjectionFunctor::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return 0;
    }

    /////////////////////////////////////////////////////////////
    // Projection Function
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProjectionFunction::ProjectionFunction(
        ProjectionID pid, ProjectionFunctor* func)
      : depth(func->get_depth()), is_exclusive(func->is_exclusive()),
        is_functional(func->is_functional()),
        is_invertible(func->is_invertible()), projection_id(pid), functor(func)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ProjectionFunction::~ProjectionFunction(void)
    //--------------------------------------------------------------------------
    {
      delete functor;
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::prepare_for_shutdown(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    LogicalRegion ProjectionFunction::project_point(
        Task* task, unsigned idx, const Domain& launch_domain,
        const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement& req = task->regions[idx];
      legion_assert(req.handle_type != LEGION_SINGULAR_PROJECTION);
      size_t arglen = 0;
      const void* args = req.get_projection_args(&arglen);
      if (!is_exclusive)
      {
        AutoLock p_lock(projection_reservation);
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          LogicalRegion result =
              !is_functional ?
                  functor->project(task, idx, req.partition, point) :
              (args == nullptr) ?
                  functor->project(req.partition, point, launch_domain) :
                  functor->project(
                      req.partition, point, launch_domain, args, arglen);
          check_projection_partition_result(req.partition, task, idx, result);
          return result;
        }
        else
        {
          LogicalRegion result =
              !is_functional ?
                  functor->project(task, idx, req.region, point) :
              (args == nullptr) ?
                  functor->project(req.region, point, launch_domain) :
                  functor->project(
                      req.region, point, launch_domain, args, arglen);
          check_projection_region_result(req.region, task, idx, result);
          return result;
        }
      }
      else
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          LogicalRegion result =
              !is_functional ?
                  functor->project(task, idx, req.partition, point) :
              (args == nullptr) ?
                  functor->project(req.partition, point, launch_domain) :
                  functor->project(
                      req.partition, point, launch_domain, args, arglen);
          check_projection_partition_result(req.partition, task, idx, result);
          return result;
        }
        else
        {
          LogicalRegion result =
              !is_functional ?
                  functor->project(task, idx, req.region, point) :
              (args == nullptr) ?
                  functor->project(req.region, point, launch_domain) :
                  functor->project(
                      req.region, point, launch_domain, args, arglen);
          check_projection_region_result(req.region, task, idx, result);
          return result;
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::project_points(
        const RegionRequirement& req, unsigned index,
        const Domain& launch_domain, const std::vector<PointTask*>& point_tasks,
        const std::vector<PointwiseDependence>* pointwise_dependences,
        const size_t total_shards, bool replaying)
    //--------------------------------------------------------------------------
    {
      legion_assert(req.handle_type != LEGION_SINGULAR_PROJECTION);
      size_t arglen = 0;
      const void* args = req.get_projection_args(&arglen);
      std::map<LogicalRegion, std::vector<DomainPoint> > dependences;
      std::vector<LogicalRegion> pointwise_regions;
      // We used to support the case of the identity projection function
      // on logical regions special with the premap case, but it is really
      // just another case of having dependences between points on a region
      // requirement so we'll detect that case that specially and handle
      // it here inside the runtime since we control the implementation of
      // the identity projection function
      const bool find_dependences =
          (IS_WRITE(req) || IS_OUTPUT_DISCARD(req)) && !IS_COLLECTIVE(req) &&
          !replaying &&
          (is_invertible || ((projection_id == 0) &&
                             (req.handle_type == LEGION_REGION_PROJECTION)));
      // Can skip pointwise analysis if we're replaying
      if (replaying)
        pointwise_dependences = nullptr;
      if (find_dependences || (pointwise_dependences != nullptr))
        pointwise_regions.reserve(point_tasks.size());
      if (!is_exclusive)
      {
        AutoLock p_lock(projection_reservation);
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          for (PointTask* const & point_task : point_tasks)
          {
            LogicalRegion result =
                !is_functional ?
                    functor->project(
                        point_task, index, req.partition,
                        point_task->get_domain_point()) :
                (args == nullptr) ?
                    functor->project(
                        req.partition, point_task->get_domain_point(),
                        launch_domain) :
                    functor->project(
                        req.partition, point_task->get_domain_point(),
                        launch_domain, args, arglen);
            check_projection_partition_result(
                req.partition, static_cast<Task*>(point_task), index, result);
            point_task->set_projection_result(index, result);

            if (find_dependences)
            {
              pointwise_regions.emplace_back(result);
              std::vector<DomainPoint>& region_deps = dependences[result];
              if (region_deps.empty())
              {
                functor->invert(
                    result, req.partition, launch_domain, region_deps);
                check_inversion(point_task, index, region_deps, launch_domain);
              }
              check_containment((point_task), index, region_deps);
            }
            else if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);
          }
        }
        else
        {
          for (PointTask* const & point_task : point_tasks)
          {
            LogicalRegion result =
                !is_functional ? functor->project(
                                     point_task, index, req.region,
                                     point_task->get_domain_point()) :
                (args == nullptr) ?
                                 functor->project(
                                     req.region, point_task->get_domain_point(),
                                     launch_domain) :
                                 functor->project(
                                     req.region, point_task->get_domain_point(),
                                     launch_domain, args, arglen);
            check_projection_region_result(
                req.region, static_cast<Task*>(point_task), index, result);
            point_task->set_projection_result(index, result);

            if (find_dependences)
            {
              pointwise_regions.emplace_back(result);
              std::vector<DomainPoint>& region_deps = dependences[result];
              if (region_deps.empty())
              {
                functor->invert(result, req.region, launch_domain, region_deps);
                check_inversion(point_task, index, region_deps, launch_domain);
              }
              check_containment((point_task), index, region_deps);
            }
            else if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);
          }
        }
      }
      else
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          for (PointTask* const & point_task : point_tasks)
          {
            LogicalRegion result =
                !is_functional ?
                    functor->project(
                        point_task, index, req.partition,
                        point_task->get_domain_point()) :
                (args == nullptr) ?
                    functor->project(
                        req.partition, point_task->get_domain_point(),
                        launch_domain) :
                    functor->project(
                        req.partition, point_task->get_domain_point(),
                        launch_domain, args, arglen);
            check_projection_partition_result(
                req.partition, static_cast<Task*>(point_task), index, result);
            point_task->set_projection_result(index, result);

            if (find_dependences)
            {
              pointwise_regions.emplace_back(result);
              std::vector<DomainPoint>& region_deps = dependences[result];
              if (region_deps.empty())
              {
                functor->invert(
                    result, req.partition, launch_domain, region_deps);
                check_inversion(point_task, index, region_deps, launch_domain);
              }
              check_containment((point_task), index, region_deps);
            }
            else if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);
          }
        }
        else
        {
          for (PointTask* const & point_task : point_tasks)
          {
            LogicalRegion result =
                !is_functional ? functor->project(
                                     point_task, index, req.region,
                                     point_task->get_domain_point()) :
                (args == nullptr) ?
                                 functor->project(
                                     req.region, point_task->get_domain_point(),
                                     launch_domain) :
                                 functor->project(
                                     req.region, point_task->get_domain_point(),
                                     launch_domain, args, arglen);
            check_projection_region_result(
                req.region, static_cast<Task*>(point_task), index, result);
            point_task->set_projection_result(index, result);

            if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);

            if (find_dependences)
            {
              pointwise_regions.emplace_back(result);
              std::vector<DomainPoint>& region_deps = dependences[result];
              if (region_deps.empty())
              {
                functor->invert(result, req.region, launch_domain, region_deps);
                check_inversion(point_task, index, region_deps, launch_domain);
              }
              check_containment((point_task), index, region_deps);
            }
            else if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);
          }
        }
      }
      if (!pointwise_regions.empty())
      {
        if (find_dependences)
        {
          // Record these now that we're no longer holding the lock
          for (unsigned idx = 0; idx < pointwise_regions.size(); idx++)
          {
            std::map<LogicalRegion, std::vector<DomainPoint> >::const_iterator
                finder = dependences.find(pointwise_regions[idx]);
            legion_assert(finder != dependences.end());
            point_tasks[idx]->record_intra_space_dependences(
                index, finder->second);
          }
        }
        if (pointwise_dependences != nullptr)
        {
          for (const PointwiseDependence& pit : *pointwise_dependences)
          {
            dependences.clear();
            // Careful! Not using the same region requirement as was originally
            // used for this projection but it has the same upper bound
            // which is good enough for us
            pit.find_dependences(req, pointwise_regions, dependences);
            for (unsigned idx = 0; idx < pointwise_regions.size(); idx++)
            {
              std::map<LogicalRegion, std::vector<DomainPoint> >::const_iterator
                  finder = dependences.find(pointwise_regions[idx]);
              if (finder == dependences.end())
                continue;
              if (total_shards > 1)
              {
                const Domain shard_domain =
                    (pit.sharding_domain == nullptr) ?
                        launch_domain :
                        pit.sharding_domain->get_tight_domain();
                for (const DomainPoint& point : finder->second)
                {
                  ShardID shard =
                      pit.sharding->shard(point, shard_domain, total_shards);
                  point_tasks[idx]->record_pointwise_dependence(
                      pit.context_index, point, shard);
                }
              }
              else
              {
                for (const DomainPoint& point : finder->second)
                  point_tasks[idx]->record_pointwise_dependence(
                      pit.context_index, point, 0 /*shard ID*/);
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::project_points(
        Operation* op, unsigned index, const RegionRequirement& req,
        const Domain& launch_domain,
        const std::vector<ProjectionPoint*>& points,
        const std::vector<PointwiseDependence>* pointwise_dependences,
        const size_t total_shards, bool replaying)
    //--------------------------------------------------------------------------
    {
      Mappable* mappable = op->get_mappable();
      legion_assert(req.handle_type != LEGION_SINGULAR_PROJECTION);
      legion_assert(mappable != nullptr);
      size_t arglen = 0;
      const void* args = req.get_projection_args(&arglen);
      const bool find_dependences =
          !replaying && is_invertible && IS_WRITE(req);
      std::map<LogicalRegion, std::vector<DomainPoint> > dependences;
      std::vector<LogicalRegion> pointwise_regions;
      // Can skip pointwise analysis if we're replaying
      if (replaying)
        pointwise_dependences = nullptr;
      if (find_dependences || (pointwise_dependences != nullptr))
        pointwise_regions.reserve(points.size());
      if (!is_exclusive)
      {
        AutoLock p_lock(projection_reservation);
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          for (ProjectionPoint* point : points)
          {
            LogicalRegion result =
                !is_functional ? functor->project(
                                     mappable, index, req.partition,
                                     point->get_domain_point()) :
                (args == nullptr) ?
                                 functor->project(
                                     req.partition, point->get_domain_point(),
                                     launch_domain) :
                                 functor->project(
                                     req.partition, point->get_domain_point(),
                                     launch_domain, args, arglen);
            check_projection_partition_result(req.partition, op, index, result);
            point->set_projection_result(index, result);

            if (find_dependences)
            {
              pointwise_regions.emplace_back(result);
              std::vector<DomainPoint>& region_deps = dependences[result];
              if (region_deps.empty())
              {
                functor->invert(
                    result, req.partition, launch_domain, region_deps);
                check_inversion(point, index, region_deps, launch_domain);
              }
              check_containment(point, index, region_deps);
            }
            else if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);
          }
        }
        else
        {
          for (ProjectionPoint* point : points)
          {
            LogicalRegion result =
                !is_functional ?
                    functor->project(
                        mappable, index, req.region,
                        point->get_domain_point()) :
                (args == nullptr) ?
                    functor->project(
                        req.region, point->get_domain_point(), launch_domain) :
                    functor->project(
                        req.region, point->get_domain_point(), launch_domain,
                        args, arglen);
            check_projection_region_result(req.region, op, index, result);
            point->set_projection_result(index, result);

            if (find_dependences)
            {
              pointwise_regions.emplace_back(result);
              std::vector<DomainPoint>& region_deps = dependences[result];
              if (region_deps.empty())
              {
                functor->invert(result, req.region, launch_domain, region_deps);
                check_inversion(point, index, region_deps, launch_domain);
              }
              check_containment(point, index, region_deps);
            }
            else if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);
          }
        }
      }
      else
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          for (ProjectionPoint* point : points)
          {
            LogicalRegion result =
                !is_functional ? functor->project(
                                     mappable, index, req.partition,
                                     point->get_domain_point()) :
                (args == nullptr) ?
                                 functor->project(
                                     req.partition, point->get_domain_point(),
                                     launch_domain) :
                                 functor->project(
                                     req.partition, point->get_domain_point(),
                                     launch_domain, args, arglen);
            check_projection_partition_result(req.partition, op, index, result);
            point->set_projection_result(index, result);

            if (find_dependences)
            {
              pointwise_regions.emplace_back(result);
              std::vector<DomainPoint>& region_deps = dependences[result];
              if (region_deps.empty())
              {
                functor->invert(
                    result, req.partition, launch_domain, region_deps);
                check_inversion(point, index, region_deps, launch_domain);
              }
              check_containment(point, index, region_deps);
            }
            else if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);
          }
        }
        else
        {
          for (ProjectionPoint* point : points)
          {
            LogicalRegion result =
                !is_functional ?
                    functor->project(
                        mappable, index, req.region,
                        point->get_domain_point()) :
                (args == nullptr) ?
                    functor->project(
                        req.region, point->get_domain_point(), launch_domain) :
                    functor->project(
                        req.region, point->get_domain_point(), launch_domain,
                        args, arglen);
            check_projection_region_result(req.region, op, index, result);
            point->set_projection_result(index, result);

            if (find_dependences)
            {
              pointwise_regions.emplace_back(result);
              std::vector<DomainPoint>& region_deps = dependences[result];
              if (region_deps.empty())
              {
                functor->invert(result, req.region, launch_domain, region_deps);
                check_inversion(point, index, region_deps, launch_domain);
              }
              check_containment(point, index, region_deps);
            }
            else if (pointwise_dependences != nullptr)
              pointwise_regions.emplace_back(result);
          }
        }
      }
      if (!pointwise_regions.empty())
      {
        if (find_dependences)
        {
          // Record these now that we're no longer holding the lock
          for (unsigned idx = 0; idx < pointwise_regions.size(); idx++)
          {
            std::map<LogicalRegion, std::vector<DomainPoint> >::const_iterator
                finder = dependences.find(pointwise_regions[idx]);
            legion_assert(finder != dependences.end());
            points[idx]->record_intra_space_dependences(index, finder->second);
          }
        }
        if (pointwise_dependences != nullptr)
        {
          for (const PointwiseDependence& pit : *pointwise_dependences)
          {
            dependences.clear();
            // Careful! Not using the same region requirement as was originally
            // used for this projection but it has the same upper bound
            // which is good enough for us
            pit.find_dependences(req, pointwise_regions, dependences);
            for (unsigned idx = 0; idx < pointwise_regions.size(); idx++)
            {
              std::map<LogicalRegion, std::vector<DomainPoint> >::const_iterator
                  finder = dependences.find(pointwise_regions[idx]);
              legion_assert(finder != dependences.end());
              if (total_shards > 1)
              {
                const Domain shard_domain =
                    (pit.sharding_domain == nullptr) ?
                        launch_domain :
                        pit.sharding_domain->get_tight_domain();
                for (const DomainPoint& point : finder->second)
                {
                  ShardID shard =
                      pit.sharding->shard(point, shard_domain, total_shards);
                  points[idx]->record_pointwise_dependence(
                      pit.context_index, point, shard);
                }
              }
              else
              {
                for (const DomainPoint& point : finder->second)
                  points[idx]->record_pointwise_dependence(
                      pit.context_index, point, 0 /*Shard ID*/);
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::find_inversions(
        OpKind op_kind, UniqueID uid, unsigned region_index,
        const RegionRequirement& req, IndexSpaceNode* domain,
        const std::vector<LogicalRegion>& points,
        std::map<LogicalRegion, std::vector<DomainPoint> >& dependences)
    //--------------------------------------------------------------------------
    {
      const Domain launch_domain = domain->get_tight_domain();
      if (!is_exclusive)
      {
        AutoLock p_lock(projection_reservation);
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          for (const LogicalRegion& point : points)
          {
            if (dependences.find(point) != dependences.end())
              continue;
            std::vector<DomainPoint>& region_deps = dependences[point];
            functor->invert(point, req.partition, launch_domain, region_deps);
            check_inversion(
                op_kind, uid, region_index, region_deps, launch_domain,
                true /*allow empty*/);
          }
        }
        else
        {
          for (const LogicalRegion& point : points)
          {
            if (dependences.find(point) != dependences.end())
              continue;
            std::vector<DomainPoint>& region_deps = dependences[point];
            functor->invert(point, req.region, launch_domain, region_deps);
            check_inversion(
                op_kind, uid, region_index, region_deps, launch_domain,
                true /*allow empty*/);
          }
        }
      }
      else
      {
        if (req.handle_type == LEGION_PARTITION_PROJECTION)
        {
          for (const LogicalRegion& point : points)
          {
            if (dependences.find(point) != dependences.end())
              continue;
            std::vector<DomainPoint>& region_deps = dependences[point];
            functor->invert(point, req.partition, launch_domain, region_deps);
            check_inversion(
                op_kind, uid, region_index, region_deps, launch_domain,
                true /*allow empty*/);
          }
        }
        else
        {
          for (const LogicalRegion& point : points)
          {
            if (dependences.find(point) != dependences.end())
              continue;
            std::vector<DomainPoint>& region_deps = dependences[point];
            functor->invert(point, req.region, launch_domain, region_deps);
            check_inversion(
                op_kind, uid, region_index, region_deps, launch_domain,
                true /*allow empty*/);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_projection_region_result(
        LogicalRegion upper_bound, const Task* task, unsigned idx,
        LogicalRegion result) const
    //--------------------------------------------------------------------------
    {
      // NO_REGION is always an acceptable answer
      if (result == LogicalRegion::NO_REGION)
        return;
      if (result.get_tree_id() != upper_bound.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Projection functor " << projection_id << " produced an "
              << "invalid logical subregion of tree ID " << result.get_tree_id()
              << " for region requirement " << idx << " of " << *task
              << " which is different from the upper bound node of tree ID "
              << upper_bound.get_tree_id() << ".";
        error.raise();
      }
      if (runtime->safe_model)
      {
        if (!runtime->is_subregion(result, upper_bound))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "Projection functor " << projection_id << " produced "
              << "an invalid logical subregion which is not a subregion of the "
              << "upper bound region for region requirement " << idx << " of "
              << *task << ".";
          error.raise();
        }
        const unsigned projection_depth =
            runtime->get_projection_depth(result, upper_bound);
        if (projection_depth > functor->get_depth())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Projection functor " << projection_id << " produced an "
                << "invalid logical subregion which has projection depth "
                << projection_depth << " which is different from stated "
                << "projection depth of the functor which is "
                << functor->get_depth() << " for region requirement " << idx
                << " of " << *task << ".";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_projection_partition_result(
        LogicalPartition upper_bound, const Task* task, unsigned idx,
        LogicalRegion result) const
    //--------------------------------------------------------------------------
    {
      // NO_REGION is always an acceptable answer
      if (result == LogicalRegion::NO_REGION)
        return;
      if (result.get_tree_id() != upper_bound.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Projection functor " << projection_id << " produced an "
              << "invalid logical subregion of tree ID " << result.get_tree_id()
              << " for region requirement " << idx << " of " << *task
              << " which is different from the upper bound node of tree ID "
              << upper_bound.get_tree_id() << ".";
        error.raise();
      }
      if (runtime->safe_model)
      {
        if (!runtime->is_subregion(result, upper_bound))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Projection functor " << projection_id << " produced an "
                << "invalid logical subregion which is not a subregion of the "
                << "upper bound region for region requirement " << idx << " of "
                << *task << ".";
          error.raise();
        }
        const unsigned projection_depth =
            runtime->get_projection_depth(result, upper_bound);
        if (projection_depth > functor->get_depth())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Projection functor " << projection_id << " produced an "
                << "invalid logical subregion which has projection depth "
                << projection_depth << " which is different from stated "
                << "projection depth of the functor which is "
                << functor->get_depth() << " for region requirement " << idx
                << " of " << *task << ".";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_projection_region_result(
        LogicalRegion upper_bound, Operation* op, unsigned idx,
        LogicalRegion result) const
    //--------------------------------------------------------------------------
    {
      // NO_REGION is always an acceptable answer
      if (result == LogicalRegion::NO_REGION)
        return;
      if (result.get_tree_id() != upper_bound.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Projection functor " << projection_id << " produced an "
              << "invalid logical subregion of tree ID " << result.get_tree_id()
              << " for region requirement " << idx << " of " << *op
              << " which is different from the upper bound node of tree ID "
              << upper_bound.get_tree_id() << ".";
        error.raise();
      }
      if (runtime->safe_model)
      {
        if (!runtime->is_subregion(result, upper_bound))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Projection functor " << projection_id << " produced an "
                << "invalid logical subregion which is not a subregion of the "
                << "upper bound region for region requirement " << idx << " of "
                << *op << ".";
          error.raise();
        }
        const unsigned projection_depth =
            runtime->get_projection_depth(result, upper_bound);
        if (projection_depth > functor->get_depth())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Projection functor " << projection_id << " produced an "
                << "invalid logical subregion which has projection depth "
                << projection_depth << " which is different from stated "
                << "projection depth of the functor which is "
                << functor->get_depth() << " for region requirement " << idx
                << " of " << *op << ".";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_projection_partition_result(
        LogicalPartition upper_bound, Operation* op, unsigned idx,
        LogicalRegion result) const
    //--------------------------------------------------------------------------
    {
      // NO_REGION is always an acceptable answer
      if (result == LogicalRegion::NO_REGION)
        return;
      if (result.get_tree_id() != upper_bound.get_tree_id())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Projection functor " << projection_id << " produced an "
              << "invalid logical subregion of tree ID " << result.get_tree_id()
              << " for region requirement " << idx << " of " << *op
              << " which is different from the upper bound node of tree ID "
              << upper_bound.get_tree_id() << ".";
        error.raise();
      }
      if (runtime->safe_model)
      {
        if (!runtime->is_subregion(result, upper_bound))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Projection functor " << projection_id << " produced an "
                << "invalid logical subregion which is not a subregion of the "
                << "upper bound region for region requirement " << idx << " of "
                << *op << ".";
          error.raise();
        }
        const unsigned projection_depth =
            runtime->get_projection_depth(result, upper_bound);
        if (projection_depth > functor->get_depth())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Projection functor " << projection_id << " produced an "
                << "invalid logical subregion which has projection depth "
                << projection_depth << " which is different from stated "
                << "projection depth of the functor which is "
                << functor->get_depth() << " for region requirement " << idx
                << " of " << *op << ".";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_inversion(
        const ProjectionPoint* point, unsigned index,
        const std::vector<DomainPoint>& points, const Domain& launch_domain,
        bool allow_empty)
    //--------------------------------------------------------------------------
    {
      const Operation* op = point->as_operation();
      if (!allow_empty && points.empty())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Projection functor " << projection_id << " produced an "
              << "empty inversion result while inverting region requirement "
              << index << " of " << *op << ". Empty inversions are never legal "
              << "because the point that produced the region must always be "
              << "included.";
        error.raise();
      }
      if (runtime->safe_model)
      {
        std::set<DomainPoint> unique_points;
        for (const DomainPoint& point : points)
        {
          if (!launch_domain.contains(point))
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Projection functor " << projection_id << " produced "
                  << "an invalid inversion result that contains points not in "
                  << "the launch domain for region requirement " << index
                  << " of " << *op << ". Only points in the launch "
                  << "domain can appear in the result.";
            error.raise();
          }
          if (!unique_points.insert(point).second)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Projection functor " << projection_id << " produced "
                  << "an invalid inversion result containing duplicate points "
                  << "for region requirement " << index << " of " << *op
                  << ". Each point is only permitted to appear once in an "
                  << "inversion.";
            error.raise();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_inversion(
        OpKind op_kind, UniqueID uid, unsigned index,
        const std::vector<DomainPoint>& points, const Domain& launch_domain,
        bool allow_empty)
    //--------------------------------------------------------------------------
    {
      if (!allow_empty && points.empty())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Projection functor " << projection_id << " produced an "
              << "empty inversion result while inverting region requirement "
              << index << " of " << Operation::get_string_rep(op_kind)
              << " (UID: " << uid << "). Empty inversions are never legal "
              << "because the point copy that produced the region must always "
              << "be included.";
        error.raise();
      }
      if (runtime->safe_model)
      {
        std::set<DomainPoint> unique_points;
        for (const DomainPoint& point : points)
        {
          if (!launch_domain.contains(point))
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Projection functor " << projection_id << " produced "
                  << "an invalid inversion result that contains points not in "
                  << "the launch domain for region requirement " << index
                  << " of " << Operation::get_string_rep(op_kind)
                  << "(UID: " << uid
                  << "). Only points in the launch domain can appear "
                  << "in the result.";
            error.raise();
          }
          if (!unique_points.insert(point).second)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Projection functor " << projection_id << " produced "
                  << "an invalid inversion result containing duplicate points "
                  << "for region requirement " << index << " of "
                  << Operation::get_string_rep(op_kind) << " (UID: " << uid
                  << "). Each point is only permitted to appear once in an "
                  << "inversion.";
            error.raise();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_containment(
        const ProjectionPoint* point, unsigned index,
        const std::vector<DomainPoint>& points)
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_model)
      {
        const DomainPoint& index_point = point->get_domain_point();
        for (const DomainPoint& point : points)
        {
          if (point == index_point)
            return;
        }
        const Operation* op = point->as_operation();
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Projection functor " << projection_id << " produced an "
              << "invalid inversion result that does not contain the original "
              << "point for region requirement " << index << " of " << *op
              << ".";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void ProjectionFunction::check_containment(
        OpKind op_kind, UniqueID uid, unsigned index,
        const DomainPoint& index_point, const std::vector<DomainPoint>& points)
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_model)
      {
        for (const DomainPoint& point : points)
        {
          if (point == index_point)
            return;
        }
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Projection functor " << projection_id << " produced an "
              << "invalid inversion result that does not contain the original "
              << "point for region requirement " << index << " of "
              << Operation::get_string_rep(op_kind) << " (UID: " << uid << ").";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    bool ProjectionFunction::is_complete(
        RegionTreeNode* node, Operation* op, unsigned index,
        IndexSpaceNode* projection_space) const
    //--------------------------------------------------------------------------
    {
      Domain launch_domain = projection_space->get_tight_domain();
      if (node->is_region())
      {
        RegionNode* region = node->as_region_node();
        if (is_functional)
        {
          if (is_exclusive)
          {
            AutoLock p_lock(projection_reservation);
            return functor->is_complete(region->handle, launch_domain);
          }
          else
            return functor->is_complete(region->handle, launch_domain);
        }
        else
        {
          Mappable* mappable = op->get_mappable();
          if (is_exclusive)
          {
            AutoLock p_lock(projection_reservation);
            return functor->is_complete(
                mappable, index, region->handle, launch_domain);
          }
          else
            return functor->is_complete(
                mappable, index, region->handle, launch_domain);
        }
      }
      else
      {
        PartitionNode* partition = node->as_partition_node();
        if (is_functional)
        {
          if (is_exclusive)
          {
            AutoLock p_lock(projection_reservation);
            return functor->is_complete(partition->handle, launch_domain);
          }
          else
            return functor->is_complete(partition->handle, launch_domain);
        }
        else
        {
          Mappable* mappable = op->get_mappable();
          if (is_exclusive)
          {
            AutoLock p_lock(projection_reservation);
            return functor->is_complete(
                mappable, index, partition->handle, launch_domain);
          }
          else
            return functor->is_complete(
                mappable, index, partition->handle, launch_domain);
        }
      }
    }

    //--------------------------------------------------------------------------
    ProjectionNode* ProjectionFunction::construct_projection_tree(
        Operation* op, unsigned index, const RegionRequirement& req,
        ShardID local_shard, RegionTreeNode* root, const ProjectionInfo& info)
    //--------------------------------------------------------------------------
    {
      ProjectionNode* result = nullptr;
      if (root->is_region())
        result = new ProjectionRegion(root->as_region_node());
      else
        result = new ProjectionPartition(root->as_partition_node());
      IndexSpaceNode* launch_space = info.projection_space;
      IndexSpace local_space =
          (info.sharding_function == nullptr) ?
              IndexSpace::NO_SPACE :
              info.sharding_function->find_shard_space(
                  local_shard, launch_space, info.sharding_space->handle,
                  op->get_provenance());
      if (!local_space.exists())
        return result;
      Domain local_domain;
      runtime->find_domain(local_space, local_domain);
      Domain launch_domain = launch_space->get_tight_domain();
      std::map<RegionTreeNode*, ProjectionNode*> node_map;
      node_map[root] = result;
      Mappable* mappable = is_functional ? nullptr : op->get_mappable();
      size_t arglen = 0;
      const void* args = req.get_projection_args(&arglen);
      if (root->is_region())
      {
        RegionNode* region = root->as_region_node();
        for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
        {
          LogicalRegion result;
          if (!is_exclusive)
          {
            AutoLock p_lock(projection_reservation);
            result =
                !is_functional ?
                    functor->project(mappable, index, region->handle, itr.p) :
                (args == nullptr) ?
                    functor->project(region->handle, itr.p, launch_domain) :
                    functor->project(
                        region->handle, itr.p, launch_domain, args, arglen);
          }
          else
            result =
                !is_functional ?
                    functor->project(mappable, index, region->handle, itr.p) :
                (args == nullptr) ?
                    functor->project(region->handle, itr.p, launch_domain) :
                    functor->project(
                        region->handle, itr.p, launch_domain, args, arglen);
          check_projection_region_result(region->handle, op, index, result);
          if (!result.exists())
            continue;
          add_to_projection_tree(result, root, node_map, local_shard);
        }
      }
      else
      {
        PartitionNode* partition = root->as_partition_node();
        for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
        {
          LogicalRegion result;
          if (!is_exclusive)
          {
            AutoLock p_lock(projection_reservation);
            result =
                !is_functional ?
                    functor->project(
                        mappable, index, partition->handle, itr.p) :
                (args == nullptr) ?
                    functor->project(partition->handle, itr.p, launch_domain) :
                    functor->project(
                        partition->handle, itr.p, launch_domain, args, arglen);
          }
          else
            result =
                !is_functional ?
                    functor->project(
                        mappable, index, partition->handle, itr.p) :
                (args == nullptr) ?
                    functor->project(partition->handle, itr.p, launch_domain) :
                    functor->project(
                        partition->handle, itr.p, launch_domain, args, arglen);
          check_projection_partition_result(
              partition->handle, op, index, result);
          if (!result.exists())
            continue;
          add_to_projection_tree(result, root, node_map, local_shard);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ProjectionFunction::add_to_projection_tree(
        LogicalRegion r, RegionTreeNode* root,
        std::map<RegionTreeNode*, ProjectionNode*>& node_map,
        ShardID owner_shard)
    //--------------------------------------------------------------------------
    {
      RegionNode* child = runtime->get_node(r);
      std::map<RegionTreeNode*, ProjectionNode*>::const_iterator finder =
          node_map.find(child);
      ProjectionRegion* current = nullptr;
      if (finder == node_map.end())
      {
        current = new ProjectionRegion(child);
        node_map[child] = current;
      }
      else
        current = static_cast<ProjectionRegion*>(finder->second);
      current->add_user(owner_shard);
      while (child != root)
      {
        // Do the next partition
        finder = node_map.find(child->parent);
        ProjectionPartition* parent = nullptr;
        if (finder == node_map.end())
        {
          parent = new ProjectionPartition(child->parent);
          node_map[child->parent] = parent;
        }
        else
          parent = static_cast<ProjectionPartition*>(finder->second);
        parent->add_child(current);
        if (child->parent == root)
          break;
        // Do the next region
        finder = node_map.find(child->parent->parent);
        ProjectionRegion* next = nullptr;
        if (finder == node_map.end())
        {
          next = new ProjectionRegion(child->parent->parent);
          node_map[child->parent->parent] = next;
        }
        else
          next = static_cast<ProjectionRegion*>(finder->second);
        next->add_child(parent);
        // Now we can walk up the tree
        child = child->parent->parent;
        current = next;
      }
    }

    /////////////////////////////////////////////////////////////
    // Cyclic Sharding Functor
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CyclicShardingFunctor::CyclicShardingFunctor(void) : ShardingFunctor()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    CyclicShardingFunctor::~CyclicShardingFunctor(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    template<int DIM>
    size_t CyclicShardingFunctor::linearize_point(
        const Realm::IndexSpace<DIM, coord_t>& is,
        const Realm::Point<DIM, coord_t>& point) const
    //--------------------------------------------------------------------------
    {
      if (is.dense())
      {
        Realm::AffineLinearizedIndexSpace<DIM, coord_t> linearizer(is);
        return linearizer.linearize(point);
      }
      else
      {
        size_t offset = 0;
        for (Realm::IndexSpaceIterator<DIM, coord_t> it(is); it.valid;
             it.step())
        {
          if (it.rect.contains(point))
          {
            Realm::AffineLinearizedIndexSpace<DIM, coord_t> linearizer(
                Realm::IndexSpace<DIM, coord_t>(it.rect));
            return offset + linearizer.linearize(point);
          }
          else
            offset += it.rect.volume();
        }
        return offset;
      }
    }

    //--------------------------------------------------------------------------
    ShardID CyclicShardingFunctor::shard(
        const DomainPoint& point, const Domain& full_space,
        const size_t total_shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(point.get_dim() == full_space.get_dim());
      switch (point.get_dim())
      {
#define DIMFUNC(DIM)                                        \
  case DIM:                                                 \
    {                                                       \
      const DomainT<DIM, coord_t> is = full_space;          \
      const Point<DIM, coord_t> p1 = point;                 \
      return (linearize_point<DIM>(is, p1) % total_shards); \
    }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          std::abort();
      }
      return 0;
    }

    /////////////////////////////////////////////////////////////
    // Sharding Function
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardingFunction::ShardingFunction(
        ShardingFunctor* func, ShardManager* m, ShardingID id, bool skip,
        bool own)
      : functor(func), manager(m), sharding_id(id),
        use_points(func->use_points()), skip_checks(skip), own_functor(own)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ShardingFunction::~ShardingFunction(void)
    //--------------------------------------------------------------------------
    {
      if (own_functor)
        delete functor;
    }

    //--------------------------------------------------------------------------
    ShardID ShardingFunction::find_owner(
        const DomainPoint& point, const Domain& sharding_space)
    //--------------------------------------------------------------------------
    {
      if (use_points)
      {
        const DomainPoint result = functor->shard_points(
            point, sharding_space, manager->shard_points,
            manager->shard_domain);
        if (manager->isomorphic_points)
        {
          if (result.get_dim() != 1)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Illegal output from sharding functor " << sharding_id
                  << ". Shards must be contained in the set of "
                  << "'shard_points' for control replicated task.";
            error.raise();
          }
          const coord_t shard = result[0];
          if (!skip_checks &&
              ((shard < 0) || (manager->total_shards <= size_t(shard))))
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Illegal output shard " << shard << " from sharding "
                  << "functor " << sharding_id
                  << ". Shards for this index space "
                  << "launch must be between 0 and " << manager->total_shards
                  << " (exclusive).";
            error.raise();
          }
          return result[0];
        }
        else
        {
          std::vector<DomainPoint>::const_iterator finder = std::lower_bound(
              manager->sorted_points.begin(), manager->sorted_points.end(),
              result);
          if (finder == manager->sorted_points.end())
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Illegal output from sharding functor " << sharding_id
                  << ". Shards must be contained in the set of "
                  << "'shard_points' for control replicated task.";
            error.raise();
          }
          const unsigned offset =
              std::distance(manager->sorted_points.begin(), finder);
          legion_assert(offset < manager->shard_lookup.size());
          return manager->shard_lookup[offset];
        }
      }
      else
      {
        const ShardID shard =
            functor->shard(point, sharding_space, manager->total_shards);
        if (!skip_checks && (manager->total_shards <= shard))
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Illegal output shard " << shard << " from sharding "
                << "functor " << sharding_id << ". Shards for this index space "
                << "launch must be between 0 and " << manager->total_shards
                << " (exclusive).";
          error.raise();
        }
        return shard;
      }
    }

    //--------------------------------------------------------------------------
    IndexSpace ShardingFunction::find_shard_space(
        ShardID shard, IndexSpaceNode* full_space, IndexSpace shard_space,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      const ShardKey key(shard, full_space->handle, shard_space);
      // Check to see if we already have it
      {
        AutoLock s_lock(sharding_lock, false /*exclusive*/);
        std::map<ShardKey, IndexSpace>::const_iterator finder =
            shard_index_spaces.find(key);
        if (finder != shard_index_spaces.end())
          return finder->second;
      }
      // Otherwise we need to make it
      IndexSpace result = full_space->create_shard_space(
          this, shard, shard_space, manager->shard_domain,
          manager->shard_points, provenance);
      AutoLock s_lock(sharding_lock);
      shard_index_spaces[key] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    bool ShardingFunction::find_shard_participants(
        IndexSpaceNode* full_space, IndexSpace shard_space,
        std::vector<ShardID>& participants)
    //--------------------------------------------------------------------------
    {
      legion_assert(participants.empty());
      std::pair<IndexSpace, IndexSpace> key(full_space->handle, shard_space);
      {
        AutoLock s_lock(sharding_lock, false /*exclusive*/);
        std::map<std::pair<IndexSpace, IndexSpace>, std::vector<ShardID> >::
            const_iterator finder = shard_participants.find(key);
        if (finder != shard_participants.end())
        {
          // If the vector is empty that means all the shards are
          // participants so we didn't need to record them all
          if (!finder->second.empty())
          {
            // Record the specific participating shards
            participants = finder->second;
            return false;
          }
          else
            return true;
        }
      }
      std::set<ShardID> range_shards;
      full_space->compute_range_shards(
          this, shard_space, manager->shard_points, manager->shard_domain,
          range_shards);
      // Should always have at least one shard participant
      legion_assert(!range_shards.empty());
      // Only need to record the results if they aren't all participating
      if (range_shards.size() < manager->total_shards)
        participants.insert(
            participants.end(), range_shards.begin(), range_shards.end());
      AutoLock s_lock(sharding_lock);
      shard_participants[key] = participants;
      return participants.empty();
    }

    //--------------------------------------------------------------------------
    bool ShardingFunction::has_participants(
        ShardID shard, IndexSpaceNode* full_space, IndexSpace shard_space)
    //--------------------------------------------------------------------------
    {
      const ShardKey key(shard, full_space->handle, shard_space);
      // Check to see if we already have it
      {
        AutoLock s_lock(sharding_lock, false /*exclusive*/);
        std::map<ShardKey, IndexSpace>::const_iterator finder =
            shard_index_spaces.find(key);
        if (finder != shard_index_spaces.end())
          return finder->second.exists();
      }
      return full_space->has_shard_participants(
          this, shard, shard_space, manager->shard_points,
          manager->shard_domain);
    }

  }  // namespace Internal
}  // namespace Legion
