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

#ifndef __LEGION_FUNCTORS_H__
#define __LEGION_FUNCTORS_H__

#include "legion/api/geometry.h"

namespace Legion {

  /**
   * \interface ProjectionFunctor
   * This defines an interface for objects that need to be
   * able to handle projection requests for an application.
   * Whenever index space tasks are launched with projection
   * region requirements, instances of this object are used
   * to handle the lowering down to individual regions for
   * specific task instances in the index space of task.
   * No more than one query of this interface will be made
   * per object at a time.
   *
   * Note also that the interface inherits from the
   * RegionTreeInspector class which gives it access to
   * all of the functions in that class for discovering
   * the shape of index space trees, field spaces, and
   * logical region trees.
   */
  class ProjectionFunctor {
  public:
    ProjectionFunctor(void);
    ProjectionFunctor(Runtime* rt);
    virtual ~ProjectionFunctor(void);
  public:
    /**
     * This is the more general implementation of projection
     * functions that work for all kinds of operations.
     * Implementations can switch on the mappable type to
     * figure out the kind of the operation that is requesting
     * the projection. The default implementation of this method
     * calls the deprecated version of this method for tasks and
     * fails for all other kinds of operations. Note that this
     * method is not passed a context, because it should only
     * be invoking context free runtime methods.
     * @param mappable the operation requesting the projection
     * @param index the index of the region requirement being projected
     * @param upper_bound the upper bound logical region
     * @param point the point being projected
     * @return logical region result
     */
    virtual LogicalRegion project(
        const Mappable* mappable, unsigned index, LogicalRegion upper_bound,
        const DomainPoint& point);
    /**
     * Same method as above, but with a partition as an upper bound
     * @param mappable the operation requesting the projection
     * @param index the index of the region requirement being projected
     * @param upper_bound the upper bound logical partition
     * @param point the point being projected
     * @return logical region result
     */
    virtual LogicalRegion project(
        const Mappable* mappable, unsigned index, LogicalPartition upper_bound,
        const DomainPoint& point);

    /**
     * This method corresponds to the one above for projecting from
     * a logical region but is only invoked if the 'is_functional'
     * method for this projection functor returns true. It must always
     * return the same result when called with the same parameters
     * @param upper_bound the upper bound logical region
     * @param point the point being projected
     * @param launch_domain the launch domain of the index operation
     * @return logical region result
     */
    virtual LogicalRegion project(
        LogicalRegion upper_bound, const DomainPoint& point,
        const Domain& launch_domain);

    /**
     * This method corresponds to the one above for projecting from
     * a logical partition but is only invoked if the 'is_functional'
     * method for this projection functor returns true. It must always
     * return the same result when called with the same parameters
     * @param upper_bound the upper bound logical partition
     * @param point the point being projected
     * @param launch_domain the launch domain of the index operation
     * @return logical region result
     */
    virtual LogicalRegion project(
        LogicalPartition upper_bound, const DomainPoint& point,
        const Domain& launch_domain);

    /**
     * This method will be invoked on functional projection functors
     * for projecting from an upper bound logical region when the
     * the corresponding region requirement has projection arguments
     * associated with it.
     * @param upper_bound the upper bound logical region
     * @param point the point being projected
     * @param launch_domain the launch domain of the index operation
     * @param args pointer to the buffer of arguments
     * @param size size of the buffer of arguments in bytes
     * @return logical region result
     */
    virtual LogicalRegion project(
        LogicalRegion upper_bound, const DomainPoint& point,
        const Domain& launch_domain, const void* args, size_t size);

    /**
     * This method will be invoked on functional projection functors
     * for projecting from an upper bound logical partition when the
     * the corresponding region requirement has projection arguments
     * associated with it.
     * @param upper_bound the upper bound logical region
     * @param point the point being projected
     * @param launch_domain the launch domain of the index operation
     * @param args pointer to the buffer of arguments
     * @param size size of the buffer of arguments in bytes
     * @return logical region result
     */
    virtual LogicalRegion project(
        LogicalPartition upper_bound, const DomainPoint& point,
        const Domain& launch_domain, const void* args, size_t size);

    /**
     * @deprecated
     * Compute the projection for a logical region projection
     * requirement down to a specific logical region.
     * @param ctx the context for this projection
     * @param task the task for the requested projection
     * @param index which region requirement we are projecting
     * @param upper_bound the upper bound logical region
     * @param point the point of the task in the index space
     * @return logical region to be used by the child task
     */
    LEGION_DEPRECATED(
        "The interface for projection functors has been "
        "updated. Please use the new 'project' methods.")
    virtual LogicalRegion project(
        Context ctx, Task* task, unsigned index, LogicalRegion upper_bound,
        const DomainPoint& point);
    /**
     * @deprecated
     * Compute the projection for a logical partition projection
     * requirement down to a specific logical region.
     * @param ctx the context for this projection
     * @param task the task for the requested projection
     * @param index which region requirement we are projecting
     * @param upper_bound the upper bound logical partition
     * @param point the point of the task in the index space
     * @return logical region to be used by the child task
     */
    LEGION_DEPRECATED(
        "The interface for projection functors has been "
        "updated. Please use the new 'project' methods.")
    virtual LogicalRegion project(
        Context ctx, Task* task, unsigned index, LogicalPartition upper_bound,
        const DomainPoint& point);
    ///@{
    /**
     * Invert the projection function. Given a logical region
     * for this operation return all of the points that alias
     * with it. Dependences will be resolved in the order that
     * they are returned to the runtime. The returned result
     * can only be empty if the region to be inverted is not
     * actually in the range of the projection given the launch
     * domain. If the region is in the range of the projection
     * then the returned result cannot be empty because it must
     * contain at least the one point that maps to the particular region.
     */
    virtual void invert(
        LogicalRegion region, LogicalRegion upper_bound,
        const Domain& launch_domain, std::vector<DomainPoint>& ordered_points);
    virtual void invert(
        LogicalRegion region, LogicalPartition upper_bound,
        const Domain& launch_domain, std::vector<DomainPoint>& ordered_points);
    ///@}

    ///@{
    /**
     * Indicate to the runtime whether this projection function
     * invoked on the given upper bound node in the region tree with
     * the given index space domain will completely "cover" the
     * all the upper bound points. Specifically will each point in
     * the upper bound node exist in at least one logical region that
     * is projected to be one of the points in the domain. It is always
     * sound to return 'false' even if the projection will ultimately
     * turn out to be complete. The only cost will be in additional
     * runtime analysis overhead. It is unsound to return 'true' if
     * the resulting projection is not complete. Undefined behavior
     * in this scenario. In general users only need to worry about
     * implementing these functions if they have a projection functor
     * that has depth greater than zero.
     * @param mappable the mappable oject for non-functional functors
     * @param index index of region requirement for non-functional functors
     * @param upper_bound the upper bound region/partition to consider
     * @param launch_domain the set of points for the projection
     * @return bool indicating whether this projection is complete
     */
    virtual bool is_complete(
        LogicalRegion upper_bound, const Domain& launch_domain);
    virtual bool is_complete(
        LogicalPartition upper_bound, const Domain& launch_domain);
    virtual bool is_complete(
        Mappable* mappable, unsigned index, LogicalRegion upper_bound,
        const Domain& launch_domain);
    virtual bool is_complete(
        Mappable* mappable, unsigned index, LogicalPartition upper_bound,
        const Domain& launch_domain);
    ///@}

    /**
     * Indicate whether calls to this projection functor
     * must be serialized or can be performed in parallel.
     * Usually they must be exclusive if this functor contains
     * state for memoizing results.
     */
    virtual bool is_exclusive(void) const { return false; }

    /*
     * Indicate whether this is a functional projection
     * functor or whether it depends on the operation being
     * launched. This will determine which project method
     * is invoked by the runtime.
     */
    virtual bool is_functional(void) const { return false; }

    /**
     * Indicate whether this is an invertible projection
     * functor which can be used to handle dependences
     * between point tasks in the same index space launch.
     */
    virtual bool is_invertible(void) const { return false; }

    /**
     * Specify the depth which this projection function goes
     * for all the points in an index space launch from
     * the upper bound node in the region tree. Depth is
     * defined as the number of levels of the region tree
     * crossed from the upper bound logical region or partition.
     * So depth 0 for a REG_PROJECTION means the same region
     * while depth 0 for a PART_PROJECTION means a subregion
     * in the immediate partition. Depth 0 is the default
     * for the identity projection function.
     */
    virtual unsigned get_depth(void) const = 0;
  private:
    friend class Internal::Runtime;
    // For pre-registered projection functors the runtime will
    // use this to initialize the runtime pointer
    inline void set_runtime(Runtime* rt) { runtime = rt; }
  protected:
    Runtime* runtime;
  };

  /**
   * \class ShardingFunctor
   *
   * A sharding functor is a object that is during control
   * replication of a task to determine which points of an
   * operation are owned by a given shard. Unlike projection
   * functors, these functors are not given access to the
   * operation being sharded. We provide access to the local
   * processor on which this operation exists and the mapping
   * of shards to processors. Legion will assume that this
   * functor is functional so the same arguments passed to
   * functor will always result in the same operation.
   */
  class ShardingFunctor {
  public:
    ShardingFunctor(void);
    virtual ~ShardingFunctor(void);
  public:
    // Indicate whether this functor wants to use the ShardID or
    // DomainPoint versions of these methods
    virtual bool use_points(void) const { return false; }
  public:
    // The ShardID version of this method
    virtual ShardID shard(
        const DomainPoint& index_point, const Domain& index_domain,
        const size_t total_shards);
    // The DomainPoint version of this method
    virtual DomainPoint shard_points(
        const DomainPoint& index_point, const Domain& index_domain,
        const std::vector<DomainPoint>& shard_points,
        const Domain& shard_domain);
  public:
    virtual bool is_invertible(void) const { return false; }
    // The ShardID version of this method
    virtual void invert(
        ShardID shard, const Domain& sharding_domain,
        const Domain& index_domain, const size_t total_shards,
        std::vector<DomainPoint>& points);
    // The DomainPoint version of this method
    virtual void invert_points(
        const DomainPoint& shard_point,
        const std::vector<DomainPoint>& shard_points,
        const Domain& shard_domain, const Domain& index_domain,
        const Domain& sharding_domain, std::vector<DomainPoint>& index_points);
  };

  /**
   * \class ConcurrentColoringFunctor
   * A concurrent coloring functor provides a functor object for
   * grouping together points in a concurrent index space task
   * launch. All the point tasks mapped by the functor to the
   * same color will be grouped together and are guaranteed
   * to execute concurrently. Point tasks mapped to different
   * colors will have no guarantee of concurrency.
   */
  class ConcurrentColoringFunctor {
  public:
    ConcurrentColoringFunctor(void);
    virtual ~ConcurrentColoringFunctor(void);
  public:
    virtual Color color(
        const DomainPoint& index_point, const Domain& index_domain) = 0;
  public:
    // You can optionally implement these methods for better performance,
    // but they are not required for correctness. If 'has_max_color'
    // returns 'true' then you must implement the 'max_color' method.
    virtual bool supports_max_color(void) { return false; }
    virtual Color max_color(const Domain& index_domain)
    {
      return std::numeric_limits<Color>::max();
    }
  };

  /**
   * \class FutureFunctor
   * A future functor object provides a callback interface
   * for applications that wants to serialize data for
   * a future only when it is absolutely necessary. Tasks
   * can return a pointer to an object that implements the
   * future functor interface. Legion will then perform
   * callbacks if/when it becomes necessary to serialize the
   * future data. If serialization is necessary then Legion will
   * perform two callbacks: first to get the future size and then
   * a second one with a buffer of that size in which to pack the
   * data. Finally, when the future is reclaimed, then Legion
   * will perform a callback to release the future functor from
   * its duties.
   */
  class FutureFunctor {
  public:
    virtual ~FutureFunctor(void) { }
  public:
    virtual const void* callback_get_future(
        size_t& size, bool& owned,
        const Realm::ExternalInstanceResource*& resource,
        void (*&freefunc)(const Realm::ExternalInstanceResource&),
        const void*& metadata, size_t& metasize) = 0;
    virtual void callback_release_future(void) = 0;
  };

  /**
   * \class PointTransformFunctor
   * A point transform functor provides a virtual function
   * infterface for transforming points in one coordinate space
   * into a different coordinate space. Calls to this functor
   * must be pure in that the same arguments passed to the
   * functor must always yield the same results.
   */
  class PointTransformFunctor {
  public:
    virtual ~PointTransformFunctor(void) { }
  public:
    virtual bool is_invertible(void) const { return false; }
    // Transform a point from the domain into a point in the range
    virtual DomainPoint transform_point(
        const DomainPoint& point, const Domain& domain,
        const Domain& range) = 0;
    // Invert a point from range and convert it into a point in the domain
    // This is only called if is_invertible returns true
    virtual DomainPoint invert_point(
        const DomainPoint& point, const Domain& domain, const Domain& range)
    {
      return DomainPoint();
    }
  };

}  // namespace Legion

#endif  // __LEGION_FUNCTORS_H__
