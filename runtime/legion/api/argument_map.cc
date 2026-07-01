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

#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/api/future_map_impl.h"
#include "legion/api/argument_map_impl.h"
#include "legion/nodes/index.h"

namespace Legion {

  /////////////////////////////////////////////////////////////
  // Argument Map
  /////////////////////////////////////////////////////////////

  //--------------------------------------------------------------------------
  ArgumentMap::ArgumentMap(void)
  //--------------------------------------------------------------------------
  {
    impl = new Internal::ArgumentMapImpl();
    legion_assert(impl != nullptr);
    impl->add_reference();
  }

  //--------------------------------------------------------------------------
  ArgumentMap::ArgumentMap(const FutureMap& rhs)
  //--------------------------------------------------------------------------
  {
    impl = new Internal::ArgumentMapImpl(rhs);
    legion_assert(impl != nullptr);
    impl->add_reference();
  }

  //--------------------------------------------------------------------------
  ArgumentMap::ArgumentMap(const ArgumentMap& rhs) : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  ArgumentMap::ArgumentMap(ArgumentMap&& rhs) noexcept : impl(rhs.impl)
  //--------------------------------------------------------------------------
  {
    rhs.impl = nullptr;
  }

  //--------------------------------------------------------------------------
  ArgumentMap::ArgumentMap(Internal::ArgumentMapImpl* i) : impl(i)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
      impl->add_reference();
  }

  //--------------------------------------------------------------------------
  ArgumentMap::~ArgumentMap(void)
  //--------------------------------------------------------------------------
  {
    if (impl != nullptr)
    {
      // Remove our reference and if we were the
      // last reference holder, then delete it
      if (impl->remove_reference())
      {
        delete impl;
      }
      impl = nullptr;
    }
  }

  //--------------------------------------------------------------------------
  ArgumentMap& ArgumentMap::operator=(const FutureMap& rhs)
  //--------------------------------------------------------------------------
  {
    // Check to see if our current impl is not nullptr,
    // if so remove our reference
    if (impl != nullptr)
    {
      if (impl->remove_reference())
      {
        delete impl;
      }
    }
    impl = new Internal::ArgumentMapImpl(rhs);
    impl->add_reference();
    return *this;
  }

  //--------------------------------------------------------------------------
  ArgumentMap& ArgumentMap::operator=(const ArgumentMap& rhs)
  //--------------------------------------------------------------------------
  {
    // Check to see if our current impl is not nullptr,
    // if so remove our reference
    if (impl != nullptr)
    {
      if (impl->remove_reference())
      {
        delete impl;
      }
    }
    impl = rhs.impl;
    // Add our reference to the new impl
    if (impl != nullptr)
    {
      impl->add_reference();
    }
    return *this;
  }

  //--------------------------------------------------------------------------
  ArgumentMap& ArgumentMap::operator=(ArgumentMap&& rhs) noexcept
  //--------------------------------------------------------------------------
  {
    if ((impl != nullptr) && impl->remove_reference())
      delete impl;
    impl = rhs.impl;
    rhs.impl = nullptr;
    return *this;
  }

  //--------------------------------------------------------------------------
  bool ArgumentMap::has_point(const DomainPoint& point)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    return impl->has_point(point);
  }

  //--------------------------------------------------------------------------
  void ArgumentMap::set_point(
      const DomainPoint& point, const UntypedBuffer& arg,
      bool replace /*=true*/)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->set_point(point, arg, replace);
  }

  //--------------------------------------------------------------------------
  void ArgumentMap::set_point(
      const DomainPoint& point, const Future& f, bool replace /*= true*/)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    impl->set_point(point, f, replace);
  }

  //--------------------------------------------------------------------------
  bool ArgumentMap::remove_point(const DomainPoint& point)
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    return impl->remove_point(point);
  }

  //--------------------------------------------------------------------------
  UntypedBuffer ArgumentMap::get_point(const DomainPoint& point) const
  //--------------------------------------------------------------------------
  {
    legion_assert(impl != nullptr);
    return impl->get_point(point);
  }

  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Argument Map Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(void)
      : Collectable(), future_map(nullptr), point_set(nullptr),
        dimensionality(0), dependent_futures(0), update_point_set(false),
        equivalent(false)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ArgumentMapImpl::ArgumentMapImpl(const FutureMap& rhs)
      : Collectable(), future_map(rhs.impl), dependent_futures(0),
        update_point_set(false), equivalent(false)
    //--------------------------------------------------------------------------
    {
      if (future_map.impl != nullptr)
      {
        point_set = future_map.impl->future_map_domain;
        point_set->add_base_expression_reference(RUNTIME_REF);
        dimensionality = point_set->get_num_dims();
      }
      else
      {
        point_set = nullptr;
        dimensionality = 0;
      }
    }

    //--------------------------------------------------------------------------
    ArgumentMapImpl::~ArgumentMapImpl(void)
    //--------------------------------------------------------------------------
    {
      if ((point_set != nullptr) &&
          point_set->remove_base_expression_reference(RUNTIME_REF))
        delete point_set;
    }

    //--------------------------------------------------------------------------
    bool ArgumentMapImpl::has_point(const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      if (dimensionality > 0)
      {
        const unsigned point_dim = point.get_dim();
        if (point_dim != dimensionality)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Mismatch in dimensionality in 'has_point' on ArgumentMap "
                << "with " << dimensionality << " dimensions and point with "
                << point_dim << " dimensions. ArgumentMaps must always contain "
                << "points of the same dimensionality.";
          error.raise();
        }
      }
      if ((point_set != nullptr) && !update_point_set &&
          point_set->contains_point(point))
        return true;
      if (future_map.impl != nullptr)
        unfreeze();
      return (arguments.find(point) != arguments.end());
    }

    //--------------------------------------------------------------------------
    void ArgumentMapImpl::set_point(
        const DomainPoint& point, const UntypedBuffer& arg, bool replace)
    //--------------------------------------------------------------------------
    {
      if (dimensionality > 0)
      {
        const unsigned point_dim = point.get_dim();
        if (point_dim != dimensionality)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Mismatch in dimensionality in 'set_point' on ArgumentMap "
                << "with " << dimensionality << " dimensions and point with "
                << point_dim << " dimensions. ArgumentMaps must always contain "
                << "points of the same dimensionality.";
          error.raise();
        }
      }
      else
      {
        dimensionality = point.get_dim();
        legion_assert(dimensionality > 0);
      }
      if (!replace and (point_set != nullptr) && !update_point_set &&
          point_set->contains_point(point))
        return;
      if (future_map.impl != nullptr)
        unfreeze();
      std::map<DomainPoint, Future>::iterator finder = arguments.find(point);
      if (finder != arguments.end())
      {
        // If it already exists and we're not replacing it then we're done
        if (!replace)
          return;
        if (finder->second.impl->producer_op != nullptr)
        {
          legion_assert(dependent_futures > 0);
          dependent_futures--;
        }
        if (arg.get_size() > 0)
          finder->second =
              Future::from_untyped_pointer(arg.get_ptr(), arg.get_size());
        else
          finder->second = Future();
      }
      else
      {
        if (arg.get_size() > 0)
          arguments[point] =
              Future::from_untyped_pointer(arg.get_ptr(), arg.get_size());
        else
          arguments[point] = Future();
        // Had to add a new point so the point set is no longer value
        update_point_set = true;
      }
      // If we modified things then they are no longer equivalent
      if (future_map.impl != nullptr)
      {
        equivalent = false;
        future_map = FutureMap();
      }
    }

    //--------------------------------------------------------------------------
    void ArgumentMapImpl::set_point(
        const DomainPoint& point, const Future& f, bool replace)
    //--------------------------------------------------------------------------
    {
      if (dimensionality > 0)
      {
        const unsigned point_dim = point.get_dim();
        if (point_dim != dimensionality)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Mismatch in dimensionality in 'set_point' on ArgumentMap "
                << "with " << dimensionality << " dimensions and point with "
                << point_dim << " dimensions. ArgumentMaps must always contain "
                << "points of the same dimensionality.";
          error.raise();
        }
      }
      else
      {
        dimensionality = point.get_dim();
        legion_assert(dimensionality > 0);
      }
      if (!replace and (point_set != nullptr) && !update_point_set &&
          point_set->contains_point(point))
        return;
      if (future_map.impl != nullptr)
        unfreeze();
      std::map<DomainPoint, Future>::iterator finder = arguments.find(point);
      if (finder != arguments.end())
      {
        // If it already exists and we're not replacing it then we're done
        if (!replace)
          return;
        if (finder->second.impl->producer_op != nullptr)
        {
          legion_assert(dependent_futures > 0);
          dependent_futures--;
        }
        finder->second = f;
      }
      else
      {
        arguments[point] = f;
        // Had to add a new point so the point set is no longer valid
        update_point_set = true;
      }
      if (f.impl->producer_op != nullptr)
        dependent_futures++;
      // If we modified things then they are no longer equivalent
      if (future_map.impl != nullptr)
      {
        equivalent = false;
        future_map = FutureMap();
      }
    }

    //--------------------------------------------------------------------------
    bool ArgumentMapImpl::remove_point(const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      if (dimensionality > 0)
      {
        const unsigned point_dim = point.get_dim();
        if (point_dim != dimensionality)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Mismatch in dimensionality in 'remove_point' on "
                << "ArgumentMap with " << dimensionality << " dimensions and "
                << "point with " << point_dim << " dimensions. ArgumentMaps "
                << "must always contain points of the same dimensionality.";
          error.raise();
        }
      }
      else
      {
        dimensionality = point.get_dim();
        legion_assert(dimensionality > 0);
      }
      if ((point_set != nullptr) && !update_point_set &&
          !point_set->contains_point(point))
        return false;
      if (future_map.impl != nullptr)
        unfreeze();
      std::map<DomainPoint, Future>::iterator finder = arguments.find(point);
      if (finder != arguments.end())
      {
        if (finder->second.impl->producer_op != nullptr)
        {
          legion_assert(dependent_futures > 0);
          dependent_futures--;
        }
        arguments.erase(finder);
        // If we modified things then they are no longer equivalent
        if (future_map.impl != nullptr)
        {
          equivalent = false;
          future_map = FutureMap();
        }
        // We removed a point so the point set is no longer valid
        update_point_set = true;
        return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    UntypedBuffer ArgumentMapImpl::get_point(const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      if (dimensionality > 0)
      {
        const unsigned point_dim = point.get_dim();
        if (point_dim != dimensionality)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error << "Mismatch in dimensionality in 'get_point' on ArgumentMap "
                << "with " << dimensionality << " dimensions and point with "
                << point_dim << " dimensions. ArgumentMaps must always contain "
                << "points of the same dimensionality.";
          error.raise();
        }
      }
      if ((point_set != nullptr) && !update_point_set &&
          !point_set->contains_point(point))
        return UntypedBuffer();
      if (future_map.impl != nullptr)
        unfreeze();
      std::map<DomainPoint, Future>::const_iterator finder =
          arguments.find(point);
      if ((finder == arguments.end()) || (finder->second.impl == nullptr))
        return UntypedBuffer();
      size_t arg_size = 0;
      const void* ptr = finder->second.impl->get_buffer(
          runtime->runtime_system_memory, &arg_size);
      return UntypedBuffer(ptr, arg_size);
    }

    //--------------------------------------------------------------------------
    FutureMap ArgumentMapImpl::freeze(InnerContext* ctx, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // If we already have a future map then we are good
      if (future_map.impl != nullptr)
        return future_map;
      // If we have no futures then we can return an empty map
      if (arguments.empty())
        return FutureMap();
      // Compute the point set if needed
      if (update_point_set)
      {
        if ((point_set != nullptr) &&
            point_set->remove_base_expression_reference(RUNTIME_REF))
          delete point_set;
        if (!arguments.empty())
        {
          Domain point_domain;
          switch (dimensionality)
          {
#define DIMFUNC(DIM)                                                     \
  case DIM:                                                              \
    {                                                                    \
      std::vector<Realm::Point<DIM, coord_t> > points(arguments.size()); \
      unsigned index = 0;                                                \
      for (const std::pair<const DomainPoint, Future>& args : arguments) \
      {                                                                  \
        const Point<DIM, coord_t> point = args.first;                    \
        points[index++] = point;                                         \
      }                                                                  \
      DomainT<DIM, coord_t> space(points);                               \
      /* Make sure this is tight for determinism */                      \
      DomainT<DIM, coord_t> tight = space.tighten();                     \
      /* Free up the sparsity map it was removed*/                       \
      if (tight.dense())                                                 \
        space.destroy();                                                 \
      point_domain = tight;                                              \
      break;                                                             \
    }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              std::abort();
          }
          IndexSpace point_space = ctx->find_index_launch_space(
              point_domain, provenance, true /*take ownership of domain*/);
          point_set = runtime->get_node(point_space);
          point_set->add_base_expression_reference(RUNTIME_REF);
        }
        else
          point_set = nullptr;
        update_point_set = false;
      }
      // See if we have any dependent future points, if we do then we need
      // to launch an explicit creation operation to ensure we get the right
      // mapping dependences for this future map
      if (point_set == nullptr)
        future_map = FutureMap();
      else if (dependent_futures == 0 && !runtime->safe_control_replication)
      {
        // Otherwise we have to make a future map and set all the futures
        // We know that they are already completed
        DistributedID did = runtime->get_available_distributed_id();
        future_map = FutureMap(new FutureMapImpl(
            ctx, point_set, did, InnerContext::NO_BLOCKING_INDEX,
            std::optional<uint64_t>(), provenance, true /*reg now*/));
        future_map.impl->set_all_futures(arguments);
      }
      else
        future_map = ctx->construct_future_map(
            point_set->handle, arguments, provenance, false /*collective*/,
            0 /*sharding id*/, true /*internal*/, false /*check space*/);
      equivalent = true;      // mark that these are equivalent
      dependent_futures = 0;  // reset this for the next unpack
      return future_map;
    }

    //--------------------------------------------------------------------------
    void ArgumentMapImpl::unfreeze(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(future_map.impl != nullptr);
      // If they are already equivalent then we're done
      if (equivalent)
        return;
      // Otherwise we need to make them equivalent
      std::map<DomainPoint, FutureImpl*> futures;
      future_map.impl->get_all_futures(futures);
      arguments.clear();
      for (const std::pair<const DomainPoint, FutureImpl*>& future : futures)
        arguments[future.first] = Future(future.second);
      if ((point_set != nullptr) &&
          point_set->remove_base_expression_reference(RUNTIME_REF))
        delete point_set;
      point_set = future_map.impl->future_map_domain;
      point_set->add_base_expression_reference(RUNTIME_REF);
      update_point_set = false;
      // Count how many dependent futures we have
      legion_assert(dependent_futures == 0);
      for (const std::pair<const DomainPoint, Future>& arg : arguments)
        if (arg.second.impl->producer_op != nullptr)
          dependent_futures++;
      equivalent = true;
    }

  }  // namespace Internal
}  // namespace Legion
